#include "tt_metal/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace blank_hlk {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t dummy;
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor transpose_hc_rm(const Tensor &a) {

    TT_ASSERT(a.shape()[3] <= 16*1024 && "transpose_hc_rm kernel doesn't support W>=16k elems yet.");
    tt_metal::Device *device = a.device();
    tt_metal::Program *program = new tt_metal::Program();
    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;
    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();
    auto ashape = a.shape();
    int N = ashape[0], C = ashape[1], H = ashape[2], W = ashape[3];

    auto bshape = ashape;
    bshape[1] = ashape[2];
    bshape[2] = ashape[1];

    TT_ASSERT(a.layout() == tt::tt_metal::Layout::ROW_MAJOR, "This transpose assumes that the data layout is row major!");

    tt_metal::Tensor output = tt_metal::Tensor(bshape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_cb_tiles = 16;
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
        program, "kernels/dataflow/transpose_hc_rm_8bank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    void *hlk_args = new blank_hlk::hlk_args_t{ .dummy = 0 };
    tt_metal::ComputeKernelArgs *blank_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(blank_hlk::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "kernels/compute/blank.cpp",
        core, blank_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

    // Compile kernels
    bool skip_hlkc = false;
    tt_metal::CompileProgram(device, program, skip_hlkc);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        binary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        dst_dram_buffer->address(),
        uint32_t(N),
        uint32_t(C),
        uint32_t(H),
        uint32_t(W),
        uint32_t(C*H)
        }
    );

    //tt_metal::WriteRuntimeArgsToDevice(device, unary_writer_kernel, core, {});

    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
