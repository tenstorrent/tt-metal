
#include "ll_buda/op_library/reshape/reshape_op.hpp"
#include "common/test_tiles.hpp"

#include "ll_buda/host_api.hpp"
#include "constants.hpp"

namespace eltwise_unary {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
}

using namespace tt::constants;

namespace tt {

namespace ll_buda {

Tensor reshape(Tensor &a, int N, int C, int H, int W) {

    if (a.layout() == Layout::TILE) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        a.reshape(N, C, H, W);
        return a;
    }

    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    // Reshape for row major data requires rebanking if and only if the last
    // dimension is changed. W dictates the stick size in DRAM banks
    if (W == a.shape()[3]) {
        a.reshape(N, C, H, W);
        return a;
    }

    TT_ASSERT(N != -1 and C != -1 and H != -1 and W != -1, "-1 reshape not yet supported for rebanking row major reshape");

    ll_buda::Program *program = new ll_buda::Program();
    tt_xy_pair core = {0, 0};

    TT_ASSERT(not a.on_host(), "Operand to reshape needs to be on device");
    TT_ASSERT(a.buffer() != nullptr, "Operand to reshape needs to be allocated in a buffer on device!");
    TT_ASSERT(a.volume() == N*C*H*W, "New shape volume must match old shape volume");

     // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    ll_buda::Tensor output = ll_buda::Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::ROW_MAJOR, device);
    ll_buda::DramBuffer *src0_dram_buffer = a.buffer();
    ll_buda::DramBuffer *dst_dram_buffer = output.buffer();

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat
    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = (a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW);
    auto cb_src0 = ll_buda::CreateCircularBuffer(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = (C * H * W / TILE_HW);
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    ll_buda::DataMovementKernel *unary_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    // No compute required, so using blank kernel
    void *hlk_args = new eltwise_unary::hlk_args_t{
        .per_core_block_cnt = int(a.volume() / TILE_HW),
        .per_core_block_size = 1
    };
    ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(eltwise_unary::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/eltwise_copy.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Compile kernels
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);
    ll_buda::ConfigureDeviceWithProgram(device, program);

    uint32_t num_old_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t old_stick_size = a.shape()[3] * 2; // Assuming bfloat16 data format
    ll_buda::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(num_old_sticks),
        uint32_t(old_stick_size),
        uint32_t(log2(old_stick_size)) }
    );

    uint32_t num_new_sticks = N*C*H;
    uint32_t new_stick_size = W * 2; // Assuming bfloat16 data format
    ll_buda::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        uint32_t(num_new_sticks),
        uint32_t(new_stick_size),
        uint32_t(log2(new_stick_size))}
    );

    ll_buda::LaunchKernels(device, program);

    output.reshape(N, C, H, W);
    delete program;
    return output;
}

} // namesapce ll_buda
} // namespace tt
