#include "libs/tt_dnn/op_library/softmax/softmax_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>

namespace hlk_softmax {
struct hlk_args_t {
    // per-batch params
    int NC;
    int Ht; // number of tiles in H to expect (expected to be a full tensor by this kernel)
    int Wt; // number of tiles in W to expect (can be a partial tensor), always <= DSTt
};
}

using u32 = std::uint32_t;
using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

extern bool GetSkipHlkc();

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

    uint32_t single_tile_size = 2 * 1024;

    Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume()/TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    Device *device = a.device();

    std::array<uint32_t, 4> output_shape = {shape[0], shape[1], W, H};
    Tensor output = Tensor(output_shape, a.dtype(), Layout::TILE, device);

    Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t in0_cb_addr  = 200 * 1024; // 2 tiles
    uint32_t in2_cb_addr  = 205 * 1024; // 2 tiles - scaler
    uint32_t out0_cb_addr = 210 * 1024; // 2 tiles
    uint32_t im0_cb_addr  = 220 * 1024; // 32 tiles
    uint32_t im1_cb_addr  = 320 * 1024; // 32 tiles
    uint32_t im2_cb_addr  = 430 * 1024; // 2 tiles (intermediates for reduce and inverse)
    uint32_t im3_cb_addr  = 440 * 1024; // 1 tile (zeros)

    // These tile capacity counts for CBs need to match the number of tiles expected by the kernel (softmax.cpp)
    uint32_t in0_t = 2, out0_t = 2;
    uint32_t im0_t = 32, im1_t = 32, im2_t = 2, im3_t = 2;
    TT_ASSERT(W < TILE_WIDTH*im0_t && "W exceeds the maximum supported size of tile buffer (kernel limitation right now).");
    // see softmax.cpp for which buffers are needed
    CreateCircularBuffer( program, device, CB::c_in0,       core, in0_t,  in0_t*single_tile_size,  in0_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_in2,       core, in0_t,  in0_t*single_tile_size,  in2_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_out0,      core, out0_t, out0_t*single_tile_size, out0_cb_addr, DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed0, core, im0_t,  im0_t*single_tile_size,  im0_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed1, core, im1_t,  im1_t*single_tile_size,  im1_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed2, core, im2_t,  im2_t*single_tile_size,  im2_cb_addr,  DataFormat::Float16_b );
    CreateCircularBuffer( program, device, CB::c_intermed3, core, im3_t,  im3_t*single_tile_size,  im3_cb_addr,  DataFormat::Float16_b );

    DataMovementKernel *reader_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/reader_unary_8bank.cpp", core,
        DataMovementProcessor::RISCV_1, NOC::RISCV_1_default);

    DataMovementKernel *writer_kernel = CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/writer_unary_8bank.cpp", core,
        DataMovementProcessor::RISCV_0, NOC::RISCV_0_default);

    vector<uint32_t> compute_args = { NC, Ht, Wt };
    ComputeKernelArgs *softmax_args = InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
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

    CompileProgram(device, program);
    ConfigureDeviceWithProgram(device, program);
    WriteRuntimeArgsToDevice(device, reader_kernel, core, { src0_dram_buffer->address(), 0, 0, num_tensor_tiles, 0,0,0,0,0 }); // [8] is scaler
    WriteRuntimeArgsToDevice(device, writer_kernel, core, { dst_dram_buffer->address(), 0, 0, num_tensor_tiles });
    LaunchKernels(device, program);

    delete program;

    return output;
} // softmax

}  // namespace ll_buda

}  // namespace tt
