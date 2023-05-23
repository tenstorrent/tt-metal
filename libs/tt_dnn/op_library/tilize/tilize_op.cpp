#include <math.h>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"

#include "llrt/tt_debug_print_server.hpp"
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
    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * 2; // Assuming bfloat16 dataformat
    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = a.shape();
    if(a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {a.shape()[0], a.shape()[2], a.shape()[3], a.shape()[1]};
    }
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = stick_s / 32;

    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = stick_s / 32;

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size};
    KernelArgs compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::KernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::KernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_sticks / 32), // per_core_block_cnt
        uint32_t(stick_s / 32) // per_core_block_tile_cnt
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
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
    tt_metal::LaunchKernels(device, program);

    return output;
}

Tensor tilize_with_zero_padding(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        std::cout << "Perf warning: tilize called on already tilized tensor." << std::endl;
        return a;
    } else {
        TT_ASSERT(a.layout() == Layout::ROW_MAJOR or a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    }
    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = a.shape();
    if(a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {a.shape()[0], a.shape()[2], a.shape()[3], a.shape()[1]};
    }
    // pad height
    output_shape[2] = (uint32_t) (ceil((double) output_shape[2] / (double) TILE_HEIGHT ) * TILE_HEIGHT);
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);


    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(output.volume() % TILE_HW == 0);
    int32_t num_tiles = output.volume() / TILE_HW;
    uint32_t row_size_datum =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_rows =  a.layout() == Layout::ROW_MAJOR ? a.shape()[2] : a.shape()[3];
    uint32_t num_rows_padded = ceil((double) num_rows / (double) TILE_HEIGHT) * TILE_HEIGHT;
    assert(row_size_datum % TILE_WIDTH == 0);
    uint32_t num_2d_faces = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] : a.shape()[0] * a.shape()[2];
    uint32_t row_size_bytes = row_size_datum * 2; // Assuming bfloat16 dataformat

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = row_size_datum / 32;
    assert(num_input_tiles > 0);
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
    uint32_t num_output_tiles = row_size_datum / 32;

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
    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto zero_buffer_l1 = new tt_metal::Buffer(device, row_size_bytes, zero_buffer_l1_addr, l1_bank_id, row_size_bytes, tt_metal::BufferType::L1);

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(row_size_bytes)) == floor(log2(row_size_bytes)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(),
                                            num_2d_faces,
                                            num_rows,
                                            num_rows_padded,
                                            row_size_bytes,
                                            zero_buffer_l1_addr};
    KernelArgs compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(row_size_bytes));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = tt_metal::KernelArgs(core, {1});
    } else {
        compile_time_args = tt_metal::KernelArgs(core, {0});
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_pad_rows.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t((num_rows_padded/TILE_HEIGHT) * num_2d_faces),
        uint32_t(row_size_datum / TILE_WIDTH)
    };

    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
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
        reader_kernel_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        (uint32_t) dram_dst_noc_xy.x,
        (uint32_t) dram_dst_noc_xy.y,
        (uint32_t) (output.shape()[0] * output.shape()[1] * output.shape()[2] * output.shape()[3] / TILE_HW)}
    );
    std::vector<uint32_t> zero_buffer_stick(row_size_datum, 0);
    tt_metal::WriteToDeviceL1(device, core, zero_buffer_l1_addr, zero_buffer_stick);
    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor tilize_conv_activation(const Tensor &a, bool conv1x1 = false) {
    TT_ASSERT(a.layout() == Layout::CHANNELS_LAST, "Conv activation should be in channels last layout");

    //vector<int> shape = {5, 4,4};
    vector<int> shape = {(int) a.shape()[1], (int) a.shape()[2], (int) a.shape()[3]};
    int R = 3;
    int S = 3;
    if(conv1x1) {
        R = 1;
        S = 1;
    }
    uint32_t num_rows = (uint32_t) (a.shape()[2] - R + 1) * (a.shape()[3] - R + 1);
    num_rows = ceil((double)num_rows / (double)TILE_HEIGHT) * TILE_HEIGHT;
    uint32_t num_cols = (uint32_t) a.shape()[1]*R*S;
    num_cols = ceil((double)num_cols / (double)TILE_WIDTH) * TILE_WIDTH;
    std::vector<uint32_t> address_map = conv_transform(shape, {R,S,1,1,0,0}, {{0,1,2},{(int) num_rows, (int) num_cols}}, 2);

    tt_metal::Program program = tt_metal::Program();
    tt_start_debug_print_server(a.device()->cluster(), {0}, {{1, 1}});

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    assert(num_cols % TILE_WIDTH == 0);
    uint32_t num_tiles_c = num_cols / TILE_WIDTH;
    uint32_t num_tiles_r = num_rows / TILE_HEIGHT;
    uint32_t num_tiles = num_tiles_r * num_tiles_c;
    uint32_t total_bytes = num_rows * num_cols * 2; // 2 for bfloat16
    uint32_t row_size = num_cols * 2; // 2 for bfloat16

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> output_shape = {1, 1, num_rows, num_cols};
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles;

    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles;

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );
    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto l1_b0_size = address_map.size() * sizeof(uint32_t);
    auto l1_b0 = tt_metal::Buffer(device, l1_b0_size, l1_bank_id, l1_b0_size, tt_metal::BufferType::L1);
    uint32_t address_map_l1_addr = l1_b0.address();
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(),
                                            (uint32_t)dram_dst_noc_xy.x,
                                            (uint32_t)dram_dst_noc_xy.y,
                                            1,
                                            num_tiles,
                                            num_tiles*single_tile_size,
                                            address_map_l1_addr};

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_dtx.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles_r), // per_core_block_cnt
        uint32_t(num_tiles_c) // per_core_block_tile_cnt
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
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
    tt_metal::WriteToDeviceL1(device, core, address_map_l1_addr, address_map);

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
        (uint32_t) (output.shape()[0] * output.shape()[1] * output.shape()[2] * output.shape()[3] / TILE_HW)}
    );
    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
