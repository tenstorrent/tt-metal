#include <math.h>

#include "tt_dnn/op_library/tilize/tilize_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {


Program tilize_single_core(const Tensor &a, Tensor& output) {

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = TILE_HW * a.element_size();

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * a.element_size(); // Assuming bfloat16 dataformat

    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (2 * single_tile_size); // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();


    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;

    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;

    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Reader compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
    };

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
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_split_rows_8bank.cpp",
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
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
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

    return program;
}

void Tilize::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR or input_tensor_a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    uint32_t stick_s =  input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.shape()[3] : input_tensor_a.shape()[1];
    uint32_t num_sticks = input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.volume() / input_tensor_a.shape()[3] : input_tensor_a.volume() / input_tensor_a.shape()[1];
    uint32_t stick_size = stick_s * input_tensor_a.element_size(); // Assuming bfloat16 dataformat

    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");
}

std::vector<Shape> Tilize::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto output_shape = input_tensor_a.shape();
    if(input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {input_tensor_a.shape()[0], input_tensor_a.shape()[2], input_tensor_a.shape()[3], input_tensor_a.shape()[1]};
    }
    return {output_shape};
}

std::vector<Tensor> Tilize::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE);
}

Program Tilize::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return tilize_single_core(input_tensor_a, output_tensor);
}

Tensor tilize(const Tensor &input_tensor_a) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    return operation::run_without_autopad(Tilize(), input_tensor_a);
}

Program tilize_with_zero_padding_single_core(const Tensor &a, Tensor &output) {

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = output.shape();


    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = output.volume() / TILE_HW;
    uint32_t row_size_datum =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_rows =  a.layout() == Layout::ROW_MAJOR ? a.shape()[2] : a.shape()[3];
    uint32_t num_rows_padded = ceil((double) num_rows / (double) TILE_HEIGHT) * TILE_HEIGHT;

    uint32_t num_2d_faces = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] : a.shape()[0] * a.shape()[2];
    uint32_t row_size_bytes = row_size_datum * a.element_size(); // Assuming bfloat16 dataformat

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_size_datum / TILE_WIDTH;
    assert(num_input_tiles > 0);
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = row_size_datum / TILE_WIDTH;

    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    auto l1_bank_ids = device->bank_ids_from_logical_core(core.start);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto zero_buffer_l1 = tt_metal::Buffer(device, row_size_bytes, l1_bank_id, row_size_bytes, tt_metal::BufferType::L1);
    uint32_t zero_buffer_l1_addr = zero_buffer_l1.address();

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
    tt_metal::WriteToDeviceL1(device, core.start, zero_buffer_l1_addr, zero_buffer_stick);

    return program;
}

void TilizeWithZeroPadding::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR or input_tensor_a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");

    uint32_t stick_s =  input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.shape()[3] : input_tensor_a.shape()[1];
    TT_ASSERT(stick_s % TILE_WIDTH == 0);
    uint32_t num_sticks = input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.volume() / input_tensor_a.shape()[3] : input_tensor_a.volume() / input_tensor_a.shape()[1];
    uint32_t stick_size = stick_s * input_tensor_a.element_size(); // Assuming bfloat16 dataformat

    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");
}

std::vector<Shape> TilizeWithZeroPadding::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto output_shape = input_tensor_a.shape();
    if(input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {input_tensor_a.shape()[0], input_tensor_a.shape()[2], input_tensor_a.shape()[3], input_tensor_a.shape()[1]};
    }
    // pad height
    output_shape[2] = (uint32_t) (ceil((double) output_shape[2] / (double) TILE_HEIGHT ) * TILE_HEIGHT);
    return {output_shape};
}

std::vector<Tensor> TilizeWithZeroPadding::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE);
}

Program TilizeWithZeroPadding::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return tilize_with_zero_padding_single_core(input_tensor_a, output_tensor);
}

Tensor tilize_with_zero_padding(const Tensor &input_tensor_a) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    return operation::run_without_autopad(TilizeWithZeroPadding(), input_tensor_a);
}

Program tilize_with_val_padding(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to tilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to tilize needs to be allocated in a buffer on device!");

    auto output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

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

    const uint32_t alignment = 32;

    uint32_t num_tiles_in_row = output_shape[3] / TILE_WIDTH;
    uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - num_blocks_w_input * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (output_shape[2] - a.shape()[2]) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (output_shape[1] - a.shape()[1]) * output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks = (output_shape[0] - a.shape()[0]) * output_shape[1] * output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = a.shape()[2] - a.shape()[2] / TILE_HEIGHT * TILE_HEIGHT;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;

    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );


    uint32_t temp_buffer_size = alignment + block_row_size;

    auto l1_bank_ids = device->bank_ids_from_logical_core(core.start);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto temp_buffer_l1 = tt_metal::Buffer(device, temp_buffer_size, l1_bank_id, temp_buffer_size, tt_metal::BufferType::L1);
    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        a.shape()[0],
        padded_W_diff_blocks,
        a.shape()[1],
        padded_Z_diff_blocks,
        a.shape()[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        a.shape()[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        packed_pad_value,
        temp_buffer_l1.address(),
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
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
        "tt_metal/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp",
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
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
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

    return program;
}

void TilizeWithValPadding::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT((input_tensor_a.layout() == Layout::ROW_MAJOR), "Can only tilize row major data");

    TT_ASSERT(input_tensor_a.shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0]);
    TT_ASSERT(input_tensor_a.shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1]);
    TT_ASSERT(input_tensor_a.shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2]);
    TT_ASSERT(input_tensor_a.shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3]);
    TT_ASSERT((this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0), "On device padding only supports padding at end of dims");

    TT_ASSERT(this->output_tensor_shape[2] % TILE_HEIGHT == 0, "Output shape must be tilizable");
    TT_ASSERT(this->output_tensor_shape[3] % TILE_WIDTH == 0, "Output shape must be tilizable");
}
std::vector<Shape> TilizeWithValPadding::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return {this->output_tensor_shape};
}
std::vector<Tensor> TilizeWithValPadding::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE);
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
Program TilizeWithValPadding::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return tilize_with_val_padding(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
}

Tensor tilize_with_val_padding(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    if (input_tensor_a.layout() == Layout::TILE) {
        if (output_tensor_shape == input_tensor_a.shape()) {
            log_warning("Perf warning: tilize with padding called on already tilized tensor of target shape.");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot tilize and pad tensor that is already tilized");
        }
    }else {
        TT_ASSERT((input_tensor_a.layout() == Layout::ROW_MAJOR), "Can only tilize row major data");
    }
    return operation::run_without_autopad(TilizeWithValPadding{output_tensor_shape, input_tensor_start, pad_value}, input_tensor_a);

}

}  // namespace tt_metal

}  // namespace tt
