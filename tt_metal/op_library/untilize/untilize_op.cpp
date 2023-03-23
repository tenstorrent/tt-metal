#include <math.h>


#include "tt_metal/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace untilize {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
    int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor untilize(const Tensor &a) {

    if (a.layout() == Layout::ROW_MAJOR) {
        std::cout << "Perf warning: Trying to untilize row major data." << std::endl;
        return a;
    }

    TT_ASSERT(a.layout() == Layout::TILE, "Can only untilize tile major data");

    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to untilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to untilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * 2; // Assuming bfloat16 dataformat


    // std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = a.shape()[3] / 32;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
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
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    // Writer compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> writer_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size};
    DataMovementKernelArgs *compile_time_args;
    if (stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(stick_size));
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Untilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_sticks / 32), // per_core_block_cnt
        uint32_t(a.shape()[3] / 32) // per_core_block_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto untilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CompileProgram(device, program, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(dram_src0_noc_xy.x),
        uint32_t(dram_src0_noc_xy.y),
        uint32_t(num_tiles) }
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        num_sticks,
        stick_size}
    );

    tt_metal::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
