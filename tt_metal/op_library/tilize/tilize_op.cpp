#include <math.h>

#include "tt_metal/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include "llrt/tests/test_libs/debug_mailbox.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor tilize(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR or a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    }
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * 2; // Assuming bfloat16 dataformat
    std::cout << "stick size (datum) " << stick_s << std::endl;
    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = stick_s / 32;

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
    uint32_t num_output_tiles = stick_s / 32;

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

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size};
    DataMovementKernelArgs *compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_sticks / 32), // per_core_block_cnt
        uint32_t(stick_s / 32) // per_core_block_tile_cnt
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "kernels/compute/3T/tilize",
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
    tt_metal::CompileProgramNew(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);


    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        (uint32_t) (a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW)}
    );
    std::cout << "Launching kernels " << std::endl;
    tt_metal::LaunchKernels(device, program);
    std::cout << "Done kernels " << std::endl;

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor tilize_with_zero_padding(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR or a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    }
    tt_metal::Program *program = new tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::InterleavedDramBuffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * 2; // Assuming bfloat16 dataformat
    std::cout << "stick size (datum) " << stick_s << std::endl;
    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");

    // InterleavedDramBuffer stores buffers across multiple dram banks but reader kernel only needs the location of the first one
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates().at(0);

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::InterleavedDramBuffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    // InterleavedDramBuffer stores buffers across multiple dram banks but writer kernel only needs the location of the first one
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates().at(0);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = stick_s / 32;

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
    uint32_t num_output_tiles = stick_s / 32;

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

    uint32_t zero_buffer_l1_addr = 600 * 1024;
    auto zero_buffer_l1 = tt_metal::CreateL1Buffer(program, device, core, stick_size, zero_buffer_l1_addr);

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size, zero_buffer_l1_addr};
    DataMovementKernelArgs *compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::InitializeCompileTimeDataMovementKernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/reader_unary_stick_layout_8bank_padding.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t(num_sticks / 32),
        uint32_t(stick_s / 32)
    };
    tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "kernels/compute/3T/tilize",
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
    tt_metal::CompileProgramNew(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);


    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        (uint32_t) (a.shape()[0] * a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW)}
    );
    std::vector<uint32_t> zero_buffer_stick(stick_s, 0);
    tt_metal::WriteToDeviceL1(device, core, zero_buffer_stick, zero_buffer_l1_addr);
    std::cout << "Launching kernels " << std::endl;
    tt_metal::LaunchKernels(device, program);
    std::cout << "Done kernels " << std::endl;

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
