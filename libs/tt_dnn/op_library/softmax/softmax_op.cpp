#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

Tensor softmax(const Tensor &a) {

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

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], H, W};
    Tensor output = Tensor(output_shape, a.dtype(), Layout::TILE, device);

    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_cb_addr  = 200 * 1024, in0_t = 16;
    uint32_t out0_cb_addr = 250 * 1024, out0_t = 16;
    uint32_t im1_cb_addr  = 300 * 1024, im1_t = 16;
    uint32_t in2_cb_addr  = 350 * 1024, in2_t = 2; // scaler for reduce coming from reader
    uint32_t im2_cb_addr  = 355 * 1024, im2_t = 2; // recip result
    uint32_t im0_cb_addr  = 360 * 1024, im0_t = 128; // buffer for exps
    TT_ASSERT(im0_t % 8 == 0 && "Size of exp Wt storage buffer must be divisible by the size of block used by the reader and compute kernel.");

    TT_ASSERT(W <= TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    // see softmax.cpp for which buffers are needed
    CreateCircularBuffer( program, device, CB::c_in0,       core, in0_t,  in0_t*single_tile_size,  in0_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_out0,      core, out0_t, out0_t*single_tile_size, out0_cb_addr, DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed1, core, im1_t,  im1_t*single_tile_size,  im1_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in2,       core, in2_t,  in2_t*single_tile_size,  in2_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed2, core, im2_t,  im2_t*single_tile_size,  im2_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed0, core, im0_t,  im0_t*single_tile_size,  im0_cb_addr,  DataFormat::Float16_b );

    DataMovementKernel *reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/reader_unary_8bank_sm.cpp", core,
        DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    DataMovementKernel *writer_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/writer_unary_8bank.cpp", core,
        DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_args = { NC, Ht, Wt };
    ComputeKernelArgs *softmax_args = InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = true;
    auto eltwise_binary_kernel = CreateComputeKernel(
        program,
        "kernels/compute/softmax.cpp",
        //"kernels/compute/eltwise_copy.cpp",
        core,
        softmax_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    bool profile = false;
    CompileProgram(device, program, profile);
    ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, reader_kernel, core, { src0_dram_buffer->address(), 0, 0, num_tensor_tiles, 0, 0, 0, 0, 0x3f800000 }); // [8]=1.0f is scaler
    WriteRuntimeArgsToDevice(device, writer_kernel, core, { dst_dram_buffer->address(), 0, 0, num_tensor_tiles });
    LaunchKernels(device, program);

    if (profile)
        tt_metal::DumpDeviceProfileResults(device, program);

    delete program;

    return output;
} // softmax

}  // namespace ll_buda

}  // namespace tt
