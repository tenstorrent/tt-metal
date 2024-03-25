// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output, bool use_pack_untilize, bool fp32_dest_acc_en) {

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.volume() / a.get_legacy_shape()[-1];
    uint32_t stick_size = a.get_legacy_shape()[-1] * output.element_size();

    uint32_t stick_s = a.get_legacy_shape()[-1];
    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (input_single_tile_size + output_single_tile_size); // 2 CBs
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
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * output.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * output.element_size();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
		.set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Writer compile-time args
    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
        0
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Untilized writer
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
    };

    std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(LogOp, "Using slow untilize.");
        compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
    } else {
        log_debug(LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0 }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end, bool use_pack_untilize, bool fp32_dest_acc_en) {

    const Shape output_shape = output.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);
    DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    log_debug("untilize_with_unpadding_single_core");
    log_debug("input_cb_data_format: {}", input_cb_data_format);
    log_debug("output_cb_data_format: {}", output_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_padded_sticks = a.get_legacy_shape()[0] * a.get_legacy_shape()[1] * a.get_legacy_shape()[2];
    uint32_t num_unpadded_sticks = a.get_legacy_shape()[0] * a.get_legacy_shape()[1] * output_shape[2];
    uint32_t padded_stick_size = a.get_legacy_shape()[3] * output.element_size(); // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_shape[3] * output.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = a.get_legacy_shape()[3] / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 - L1_UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (output.element_size() * TILE_HEIGHT * 2 + output.element_size());
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
    uint32_t block_row_size = block_width * output.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - num_blocks_w_output * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (a.get_legacy_shape()[2] - output_shape[2]) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (a.get_legacy_shape()[1] - output_shape[1]) * a.get_legacy_shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (a.get_legacy_shape()[0] - output_shape[0]) * a.get_legacy_shape()[1] * a.get_legacy_shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_shape[2] - output_shape[2] / TILE_HEIGHT * TILE_HEIGHT;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
		.set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        output_shape[0],
        padded_W_diff_blocks,
        output_shape[1],
        padded_Z_diff_blocks,
        output_shape[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        output_shape[3],
        unpadded_stick_size,
        padded_stick_size,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_stick_size;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)std::log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
        (std::uint32_t) (input_cb_data_format == tt::DataFormat::Float32)
    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // Untilized writer
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/untilize/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
    };

    std::string compute_kernel("tt_eager/tt_dnn/op_library/untilize/kernels/compute/pack_untilize.cpp");
    if (num_tiles_per_block > MAX_PACK_UNTILIZE_WIDTH || !use_pack_untilize) {
        log_debug(LogOp, "Using slow untilize.");
        compute_kernel = std::string("tt_eager/tt_dnn/op_library/untilize/kernels/compute/untilize.cpp");
    } else {
        log_debug(LogOp, "Using fast pack untilize.");
    }

    auto untilize_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel,
        core,
        tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
