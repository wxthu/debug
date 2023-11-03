#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "cu_helper.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <string>

//full_dimensions: [128, 100352, 1152]

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(256) default_function_kernel(float* __restrict__ conv_unpad, float* __restrict__ data, float* __restrict__ kernel);
extern "C" __global__ void __launch_bounds__(256) default_function_kernel(float* __restrict__ conv_unpad, float* __restrict__ data, float* __restrict__ kernel) {
  float conv_local[32];
  __shared__ float data_pad_shared[512];
  __shared__ float kernel_pad_shared[1024];
  float data_pad_shared_local[4];
  float kernel_pad_shared_local[8];
  conv_local[0] = 0.000000e+00f;
  conv_local[4] = 0.000000e+00f;
  conv_local[8] = 0.000000e+00f;
  conv_local[12] = 0.000000e+00f;
  conv_local[16] = 0.000000e+00f;
  conv_local[20] = 0.000000e+00f;
  conv_local[24] = 0.000000e+00f;
  conv_local[28] = 0.000000e+00f;
  conv_local[1] = 0.000000e+00f;
  conv_local[5] = 0.000000e+00f;
  conv_local[9] = 0.000000e+00f;
  conv_local[13] = 0.000000e+00f;
  conv_local[17] = 0.000000e+00f;
  conv_local[21] = 0.000000e+00f;
  conv_local[25] = 0.000000e+00f;
  conv_local[29] = 0.000000e+00f;
  conv_local[2] = 0.000000e+00f;
  conv_local[6] = 0.000000e+00f;
  conv_local[10] = 0.000000e+00f;
  conv_local[14] = 0.000000e+00f;
  conv_local[18] = 0.000000e+00f;
  conv_local[22] = 0.000000e+00f;
  conv_local[26] = 0.000000e+00f;
  conv_local[30] = 0.000000e+00f;
  conv_local[3] = 0.000000e+00f;
  conv_local[7] = 0.000000e+00f;
  conv_local[11] = 0.000000e+00f;
  conv_local[15] = 0.000000e+00f;
  conv_local[19] = 0.000000e+00f;
  conv_local[23] = 0.000000e+00f;
  conv_local[27] = 0.000000e+00f;
  conv_local[31] = 0.000000e+00f;
  for (int ra_fused0_outer = 0; ra_fused0_outer < 144; ++ra_fused0_outer) {
    __syncthreads();
    data_pad_shared[((int)threadIdx.x)] = data[(((((((((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 49) * 115200) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 6)) / 9) * 900)) + (((((((int)blockIdx.x) * 16) + ((((int)threadIdx.x) & 63) >> 2)) % 196) / 7) * 30)) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 6)) % 9) / 3) * 30)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) & 63)) % 28)) + (((ra_fused0_outer * 2) + (((int)threadIdx.x) >> 6)) % 3))];
    data_pad_shared[(((int)threadIdx.x) + 256)] = data[(((((((((((int)blockIdx.x) * 4) + ((((int)threadIdx.x) & 63) >> 4)) / 49) * 115200) + (((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 6)) + 4) / 9) * 900)) + (((((((int)blockIdx.x) * 16) + ((((int)threadIdx.x) & 63) >> 2)) % 196) / 7) * 30)) + ((((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 6)) + 4) % 9) / 3) * 30)) + (((((int)blockIdx.x) * 8) + (((int)threadIdx.x) & 63)) % 28)) + ((((ra_fused0_outer * 8) + (((int)threadIdx.x) >> 6)) + 1) % 3))];
    kernel_pad_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 3) * 1152) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7))];
    kernel_pad_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) >> 3) * 1152) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)) + 36864)];
    kernel_pad_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 3) * 1152) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)) + 73728)];
    kernel_pad_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) >> 3) * 1152) + (ra_fused0_outer * 8)) + (((int)threadIdx.x) & 7)) + 110592)];
    __syncthreads();
    for (int ra_fused0_inner_outer = 0; ra_fused0_inner_outer < 8; ++ra_fused0_inner_outer) {
      data_pad_shared_local[0] = data_pad_shared[((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15))];
      data_pad_shared_local[1] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 16)];
      data_pad_shared_local[2] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 32)];
      data_pad_shared_local[3] = data_pad_shared[(((ra_fused0_inner_outer * 64) + (((int)threadIdx.x) & 15)) + 48)];
      kernel_pad_shared_local[0] = kernel_pad_shared[(((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer)];
      kernel_pad_shared_local[1] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 128)];
      kernel_pad_shared_local[2] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 256)];
      kernel_pad_shared_local[3] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 384)];
      kernel_pad_shared_local[4] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 512)];
      kernel_pad_shared_local[5] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 640)];
      kernel_pad_shared_local[6] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 768)];
      kernel_pad_shared_local[7] = kernel_pad_shared[((((((int)threadIdx.x) >> 4) * 8) + ra_fused0_inner_outer) + 896)];
      conv_local[0] = (conv_local[0] + (data_pad_shared_local[0] * kernel_pad_shared_local[0]));
      conv_local[4] = (conv_local[4] + (data_pad_shared_local[0] * kernel_pad_shared_local[1]));
      conv_local[8] = (conv_local[8] + (data_pad_shared_local[0] * kernel_pad_shared_local[2]));
      conv_local[12] = (conv_local[12] + (data_pad_shared_local[0] * kernel_pad_shared_local[3]));
      conv_local[16] = (conv_local[16] + (data_pad_shared_local[0] * kernel_pad_shared_local[4]));
      conv_local[20] = (conv_local[20] + (data_pad_shared_local[0] * kernel_pad_shared_local[5]));
      conv_local[24] = (conv_local[24] + (data_pad_shared_local[0] * kernel_pad_shared_local[6]));
      conv_local[28] = (conv_local[28] + (data_pad_shared_local[0] * kernel_pad_shared_local[7]));
      conv_local[1] = (conv_local[1] + (data_pad_shared_local[1] * kernel_pad_shared_local[0]));
      conv_local[5] = (conv_local[5] + (data_pad_shared_local[1] * kernel_pad_shared_local[1]));
      conv_local[9] = (conv_local[9] + (data_pad_shared_local[1] * kernel_pad_shared_local[2]));
      conv_local[13] = (conv_local[13] + (data_pad_shared_local[1] * kernel_pad_shared_local[3]));
      conv_local[17] = (conv_local[17] + (data_pad_shared_local[1] * kernel_pad_shared_local[4]));
      conv_local[21] = (conv_local[21] + (data_pad_shared_local[1] * kernel_pad_shared_local[5]));
      conv_local[25] = (conv_local[25] + (data_pad_shared_local[1] * kernel_pad_shared_local[6]));
      conv_local[29] = (conv_local[29] + (data_pad_shared_local[1] * kernel_pad_shared_local[7]));
      conv_local[2] = (conv_local[2] + (data_pad_shared_local[2] * kernel_pad_shared_local[0]));
      conv_local[6] = (conv_local[6] + (data_pad_shared_local[2] * kernel_pad_shared_local[1]));
      conv_local[10] = (conv_local[10] + (data_pad_shared_local[2] * kernel_pad_shared_local[2]));
      conv_local[14] = (conv_local[14] + (data_pad_shared_local[2] * kernel_pad_shared_local[3]));
      conv_local[18] = (conv_local[18] + (data_pad_shared_local[2] * kernel_pad_shared_local[4]));
      conv_local[22] = (conv_local[22] + (data_pad_shared_local[2] * kernel_pad_shared_local[5]));
      conv_local[26] = (conv_local[26] + (data_pad_shared_local[2] * kernel_pad_shared_local[6]));
      conv_local[30] = (conv_local[30] + (data_pad_shared_local[2] * kernel_pad_shared_local[7]));
      conv_local[3] = (conv_local[3] + (data_pad_shared_local[3] * kernel_pad_shared_local[0]));
      conv_local[7] = (conv_local[7] + (data_pad_shared_local[3] * kernel_pad_shared_local[1]));
      conv_local[11] = (conv_local[11] + (data_pad_shared_local[3] * kernel_pad_shared_local[2]));
      conv_local[15] = (conv_local[15] + (data_pad_shared_local[3] * kernel_pad_shared_local[3]));
      conv_local[19] = (conv_local[19] + (data_pad_shared_local[3] * kernel_pad_shared_local[4]));
      conv_local[23] = (conv_local[23] + (data_pad_shared_local[3] * kernel_pad_shared_local[5]));
      conv_local[27] = (conv_local[27] + (data_pad_shared_local[3] * kernel_pad_shared_local[6]));
      conv_local[31] = (conv_local[31] + (data_pad_shared_local[3] * kernel_pad_shared_local[7]));
    }
  }
  conv_unpad[((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15))] = conv_local[0];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 1605632)] = conv_local[4];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 3211264)] = conv_local[8];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 4816896)] = conv_local[12];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 6422528)] = conv_local[16];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 8028160)] = conv_local[20];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 9633792)] = conv_local[24];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 11239424)] = conv_local[28];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 16)] = conv_local[1];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 1605648)] = conv_local[5];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 3211280)] = conv_local[9];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 4816912)] = conv_local[13];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 6422544)] = conv_local[17];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 8028176)] = conv_local[21];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 9633808)] = conv_local[25];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 11239440)] = conv_local[29];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 32)] = conv_local[2];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 1605664)] = conv_local[6];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 3211296)] = conv_local[10];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 4816928)] = conv_local[14];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 6422560)] = conv_local[18];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 8028192)] = conv_local[22];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 9633824)] = conv_local[26];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 11239456)] = conv_local[30];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 48)] = conv_local[3];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 1605680)] = conv_local[7];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 3211312)] = conv_local[11];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 4816944)] = conv_local[15];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 6422576)] = conv_local[19];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 8028208)] = conv_local[23];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 9633840)] = conv_local[27];
  conv_unpad[(((((((int)threadIdx.x) >> 4) * 100352) + (((int)blockIdx.x) * 64)) + (((int)threadIdx.x) & 15)) + 11239472)] = conv_local[31];
}


int main(int argc, char *argv[])
{
    std::string path;
    int input_size0 = 14745600;
    int input_size1 = 147456;
    int output_size0 = 12845056;

    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, CU_CTX_SCHED_AUTO/*CU_CTX_SCHED_YIELD*/ | CU_CTX_MAP_HOST, device));

    float *input0h, *input1h, *output0h;
    float *input0d, *input1d, *output0d;
    input0h = (float*)malloc(58982400);
    input1h = (float*)malloc(589824);

    cudaMalloc((void **)&input0d, 58982400);
    cudaMalloc((void **)&input1d, 589824);
    cudaMalloc((void **)&output0d, 51380224);

    srand(1);
    for (int i = 0; i < input_size0; ++ i)
        input0h[i] = 1;
    for (int i = 0; i < input_size1; ++ i)
        input1h[i] = 1;

    cudaMemcpy(input0d, input0h, 58982400, cudaMemcpyHostToDevice);
    cudaMemcpy(input1d, input1h, 589824, cudaMemcpyHostToDevice);

    dim3 grid(1568, 1, 1);
    dim3 block(256, 1, 1);
    
	for (int i = 0; i < 10; ++i)
    {
        default_function_kernel<<<grid, block>>>((float*)input0d, (float*)input1d, (float*)output0d);
        cudaDeviceSynchronize();
    }
}
