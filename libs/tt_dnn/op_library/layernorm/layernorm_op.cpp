#include "libs/tt_dnn/op_library/layernorm/layernorm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

Tensor layernorm_(const Tensor &a, float eps, const Tensor* gamma, const Tensor* beta) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    u32 HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    Program *program = new Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr, "Operand to transpose_wh op needs to be on device!");
    TT_ASSERT(Wt % 8 == 0 && "Wt must be divisible by the size of block used by the reader and compute kernel.");

    uint32_t single_tile_size = 2 * 1024;

    Buffer *src0_dram_buffer = a.buffer();
    auto gamma_dram_addr = gamma ? gamma->buffer()->address() : 0;
    auto beta_dram_addr = beta ? beta->buffer()->address() : 0;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;
    uint32_t num_gamma_tiles = gamma ? gamma->volume()/TILE_HW : 0;
    uint32_t num_beta_tiles = beta ? beta->volume()/TILE_HW : 0;

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], H, W};
    Tensor output = Tensor(output_shape, a.dtype(), Layout::TILE, device);

    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t  =  64; // space for 32 tiles plus buffering 32 tiles from NC
    uint32_t out0_t =  64; // output can use less space TODO(AP)
    uint32_t im0_t  =  64; // buffer for saving xmm
    uint32_t im3_t  =  64; // buffer for xmm^2
    uint32_t in5_t  =  32; // buffer for gamma
    uint32_t in6_t  =  32; // buffer for beta
    uint32_t im5_t  =  16; // 2*block_size - for buffering to/from *gamma/+beta

    uint32_t im4_t  =   8; // 8 just in case, 4 would prob suffice
    uint32_t in4_t  =   2; // ones column mask
    uint32_t im1_t  =   2;
    uint32_t in2_t  =   2; // scaler for reduce coming from reader
    uint32_t in3_t  =   2; // epsilon coming from reader
    uint32_t im2_t  =   2; //

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    TT_ASSERT(im0_t % 8 == 0 && "Size of exp Wt storage buffer must be divisible by the size of block used by the reader and compute kernel.");

    // see softmax.cpp for which buffers are needed
    CreateCircularBuffer( program, device, CB::c_in0,       core, in0_t,  in0_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_out0,      core, out0_t, out0_t*single_tile_size, DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed1, core, im1_t,  im1_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in2,       core, in2_t,  in2_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in3,       core, in3_t,  in3_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in4,       core, in4_t,  in4_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed2, core, im2_t,  im2_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed0, core, im0_t,  im0_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed3, core, im3_t,  im3_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed4, core, im4_t,  im4_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed5, core, im5_t,  im5_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in5,       core, in5_t,  in5_t*single_tile_size,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in6,       core, in6_t,  in6_t*single_tile_size,  DataFormat::Float16_b );

    DataMovementKernel *reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/reader_unary_8bank_ln.cpp", core,
        DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    DataMovementKernel *writer_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/writer_unary_8bank.cpp", core,
        DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_args = { NC, Ht, Wt, num_gamma_tiles>0, num_beta_tiles>0 };
    ComputeKernelArgs *softmax_args = InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto eltwise_binary_kernel = CreateComputeKernel(
        program,
        "kernels/compute/layernorm.cpp",
        core,
        softmax_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    bool profile = false;
    CompileProgram(device, program, profile);
    ConfigureDeviceWithProgram(device, program);
    union { float f; uint32_t u; } winv; winv.f = 1.0f / W; // bcast-w scaler
    union { float f; uint32_t u; } e; e.f = eps; // epsilon
    uint32_t gamma_tiles = gamma ? gamma->volume() / TILE_HW : 0;
    uint32_t beta_tiles = beta ? beta->volume() / TILE_HW : 0;
    std::cout << "Num gamma=" << num_gamma_tiles << " addr=" << gamma_dram_addr << std::endl;
    std::cout << "Num beta=" << num_beta_tiles << " addr=" << beta_dram_addr << std::endl;
    WriteRuntimeArgsToDevice( device, reader_kernel, core,
        { src0_dram_buffer->address(), 0, 0, num_tensor_tiles, 0, 0, 0, 0, winv.u, e.u, // 0-9
          num_gamma_tiles, gamma_dram_addr, num_beta_tiles, beta_dram_addr } // 10-13
    );
    WriteRuntimeArgsToDevice( device, writer_kernel, core, { dst_dram_buffer->address(), 0, 0, num_tensor_tiles } );
    LaunchKernels(device, program);

    if (profile)
        tt_metal::DumpDeviceProfileResults(device, program);

    delete program;

    return output;
} // softmax

Tensor layernorm(const Tensor &a, float eps) { return layernorm_(a, eps, nullptr, nullptr); }
Tensor layernorm_gamma(const Tensor &a, float eps, const Tensor& gamma) { return layernorm_(a, eps, &gamma, nullptr); }
Tensor layernorm_gamma_beta(const Tensor &a, float eps, const Tensor& gamma, const Tensor& beta) { return layernorm_(a, eps, &gamma, &beta); }

}  // namespace ll_buda

}  // namespace tt
