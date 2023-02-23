#include <math.h>


#include "ll_buda/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "ll_buda/host_api.hpp"
#include "constants.hpp"

namespace tilize {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
    int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};
}

using namespace tt::constants;

namespace tt {

namespace ll_buda {

Tensor tilize(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Can only tilize row major data");
    }
    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");
    TT_ASSERT(a.shape()[0] == 1 && "Only N=1 is supported in tilize!");

    uint32_t single_tile_size = 2 * TILE_HW;

    ll_buda::DramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * 2; // Assuming bfloat16 dataformat

    std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates(a.device());

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    ll_buda::Tensor output = ll_buda::Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    ll_buda::DramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates(output.device());

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = a.shape()[3] / 32;

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
    uint32_t num_output_tiles = a.shape()[3] / 32;

    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    // Tilized reader
    ll_buda::DataMovementKernel *unary_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    // Tilized writer
    ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    //  ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
    //     program,
    //     "kernels/dataflow/writer_unary_8bank.cpp",
    //     core,
    //     ll_buda::DataMovementProcessor::RISCV_0,
    //     ll_buda::NOC::RISCV_0_default);

    void *hlk_args = new tilize::hlk_args_t{
        .per_core_block_cnt = int32_t(num_sticks / 32),
        .per_core_block_tile_cnt = int32_t(a.shape()[3] / 32)
    };
    ll_buda::ComputeKernelArgs *eltwise_unary_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(tilize::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = ll_buda::CreateComputeKernel(
        program,
        "kernels/compute/tilize.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    ll_buda::ConfigureDeviceWithProgram(device, program);

    ll_buda::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(num_sticks),
        uint32_t(stick_size),
        uint32_t(log2(stick_size)) }
    );

    ll_buda::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        (uint32_t) (a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW)}
    );

    // ll_buda::WriteRuntimeArgsToDevice(
    //     device,
    //     unary_writer_kernel,
    //     core,
    //     {dst_dram_buffer->address(),
    //     uint32_t(dram_dst_noc_xy.x),
    //     uint32_t(dram_dst_noc_xy.y),
    //     uint32_t(num_tiles)}
    // );

    // assert(false);
    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
