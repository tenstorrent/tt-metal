#include "tt_metal/op_library/fill_rm/fill_rm_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;
using u32 = uint32_t;

namespace tt {

namespace tt_metal {

tt_metal::Tensor fill_rm(int N, int C, int H, int W, int hFill, int wFill, const tt_metal::Tensor& any, int val_hi, int val_lo) {

    TT_ASSERT(hFill <= H && wFill <= W);
    tt_metal::Device *device = any.device();
    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};
    TT_ASSERT(val_hi >= 0 && val_hi < 0xFFFF); // TODO(AP): support dtypes..
    TT_ASSERT(val_lo >= 0 && val_lo < 0xFFFF);

    uint32_t single_tile_size = 2 * TILE_HW;
    std::array<uint32_t, 4> bshape = {u32(N),u32(C),u32(H),u32(W)};

    tt_metal::Tensor output = tt_metal::Tensor(bshape, any.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_cb_tiles = 16;
    TT_ASSERT(W < 1024*num_cb_tiles); // Limitation for simplifying the kernel
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        0, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        200*1024, // cb addr
        DataFormat::Float16_b);
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        1, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        300*1024,
        DataFormat::Float16_b);

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/fill_rm_8bank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        0 // dummy
    };
    tt_metal::ComputeKernelArgs *blank_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, blank_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

    // Compile kernels
    bool skip_hlkc = false;
    tt_metal::CompileProgram(device, program, skip_hlkc);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteRuntimeArgsToDevice(
        device, binary_reader_kernel, core,
        { dst_dram_buffer->address(), u32(N*C), u32(H), u32(W), u32(hFill), u32(wFill), u32(val_hi), u32(val_lo) }
    );

    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
