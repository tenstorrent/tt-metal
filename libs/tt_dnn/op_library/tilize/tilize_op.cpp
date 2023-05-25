#include <math.h>

#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
#include "llrt/tt_debug_print_server.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

// This function needs to be up to date with tilize to ensure accurate l1 usage calcaulation
bool check_tilize_l1_size(const Tensor &a) {
    if (a.layout() == Layout::ROW_MAJOR || a.layout() == Layout::CHANNELS_LAST) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t single_tile_size = a.element_size() * TILE_HW;
        uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
        uint32_t cb_buffers_size = 2 * (stick_s / TILE_WIDTH * single_tile_size);
        return max_l1_size >= cb_buffers_size;
    } else {
        return false;
    }
}

// This function needs to be up to date with tilize to ensure accurate l1 usage calcaulation
bool check_tilize_with_val_padding_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start) {
    if (a.layout() == Layout::ROW_MAJOR) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t single_tile_size = a.element_size() * TILE_HW;
        const uint32_t alignment = 32;
        uint32_t unpadded_row_size_bytes = a.shape()[3] * a.element_size();
        uint32_t padded_row_size_datum = output_tensor_shape[3];
        uint32_t cb_buffers_size = 2 * (padded_row_size_datum / TILE_WIDTH * single_tile_size);
        uint32_t temp_buffer_size = alignment + unpadded_row_size_bytes;
        return max_l1_size >= cb_buffers_size + temp_buffer_size;
    } else {
        return false;
    }
}

Tensor tilize(const Tensor &a) {
    if (a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
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
    std::vector<uint32_t> compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = {1};
    } else {
        compile_time_args = {0};
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

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        compute_args,
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
        log_warning("Perf warning: tilize called on already tilized tensor.");
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
    std::vector<uint32_t> compile_time_args;
    if (stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(row_size_bytes));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = {1};
    } else {
        compile_time_args = {0};
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

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        compute_kernel_args,
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

Tensor tilize_with_val_padding(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    if (a.layout() == Layout::TILE) {
        if (output_tensor_shape == a.shape()) {
            log_warning("Perf warning: tilize with padding called on already tilized tensor of target shape.");
            return a;
        } else {
            TT_ASSERT(false, "Cannot tilize and pad tensor that is already tilized");
        }
    }else {
        TT_ASSERT((a.layout() == Layout::ROW_MAJOR), "Can only tilize row major data");
    }

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");
    TT_ASSERT(a.shape()[0] + input_tensor_start[0] <= output_tensor_shape[0]);
    TT_ASSERT(a.shape()[1] + input_tensor_start[1] <= output_tensor_shape[1]);
    TT_ASSERT(a.shape()[2] + input_tensor_start[2] <= output_tensor_shape[2]);
    TT_ASSERT(a.shape()[3] + input_tensor_start[3] <= output_tensor_shape[3]);
    TT_ASSERT((input_tensor_start[0] == 0 && input_tensor_start[1] == 0 && input_tensor_start[2] == 0 && input_tensor_start[3] == 0), "On device padding only supports padding at end of dims");

    auto output_shape = output_tensor_shape;

    TT_ASSERT(output_shape[2] % TILE_HEIGHT == 0, "Output shape must be tilizable");
    TT_ASSERT(output_shape[3] % TILE_WIDTH == 0, "Output shape must be tilizable");

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = output.volume() / TILE_HW;

    uint32_t unpadded_row_size_datum = a.shape()[3];
    uint32_t padded_row_size_datum = output_shape[3];

    uint32_t num_rows_padded = output_shape[2];
    uint32_t num_cols_padded = output_shape[3] - unpadded_row_size_datum;


    uint32_t num_2d_faces = output_shape[0] * output_shape[1];

    uint32_t unpadded_row_size_bytes = unpadded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = padded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = padded_row_size_datum / TILE_WIDTH;
    assert(num_input_tiles > 0);
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
    uint32_t num_output_tiles = padded_row_size_datum / TILE_WIDTH;

    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t alignment = 32;
    uint32_t temp_buffer_size = alignment + unpadded_row_size_bytes;

    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto temp_buffer_l1 = tt_metal::Buffer(device, temp_buffer_size, l1_bank_id, temp_buffer_size, tt_metal::BufferType::L1);
    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        a.shape()[0],
        output_shape[0],
        a.shape()[1],
        output_shape[1],
        a.shape()[2],
        output_shape[2],
        a.shape()[3],
        output_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        packed_pad_value,
        temp_buffer_l1.address()
    };
    std::vector<uint32_t> compile_time_args;
    // Reader compile-time args
    // Data is 32 byte aligned
    uint32_t stick_size = unpadded_row_size_bytes;
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    if (stick_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = {1};
        reader_kernel_args.push_back((std::uint32_t)log2(stick_size));
    } else {
        compile_time_args = {0};
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_pad_dims.cpp",
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
        uint32_t((num_rows_padded / TILE_HEIGHT) * num_2d_faces),
        uint32_t(padded_row_size_datum / TILE_WIDTH)
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        compute_kernel_args,
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
        (uint32_t) num_tiles}
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
