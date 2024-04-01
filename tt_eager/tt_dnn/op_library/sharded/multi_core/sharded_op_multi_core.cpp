// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// Utility function
uint32_t calculate_starting_idx_h (const Tensor& tensor, uint32_t num_slices, uint32_t slice_index) {
    if (num_slices <= 1) {
        return 0;
    }

    uint32_t num_tiles_height = tensor.volume() / tensor.get_legacy_shape()[-1] / TILE_HEIGHT;
    uint32_t num_tiles_width = tensor.get_legacy_shape()[-1] / TILE_WIDTH;
    uint32_t total_num_tiles = num_tiles_height * num_tiles_width;

    uint32_t num_tiles_per_slice = total_num_tiles / num_slices;
    uint32_t starting_tile_in_slice = num_tiles_per_slice * slice_index;
    return starting_tile_in_slice;
}

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = output.shard_spec().value();
    auto shard_strategy = output.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end;
    if (input.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (input.volume() / input.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.get_legacy_shape()[-1] * input.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            input_unit_size - (round_up(num_units_per_row, input_unit_size) - num_units_per_row);
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    auto all_cores = shard_spec.grid;
    uint32_t input_cb_index = CB::c_in0;
    uint32_t scratch_cb_index = CB::c_in1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = align(output_unit_size, ADDRESS_ALIGNMENT);
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t input_page_size = align(input_unit_size, ADDRESS_ALIGNMENT);
        tt_metal::CircularBufferConfig input_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{input_cb_index, input_cb_data_format}})
                .set_page_size(input_cb_index, input_page_size);
        auto cb_input = tt_metal::CreateCircularBuffer(program, all_cores, input_cb_out_config);
    }
    tt_metal::CircularBufferConfig output_cb_out_config =
        tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
            .set_page_size(out_cb_index, output_page_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    if (src_is_dram && input_unit_size % DRAM_ALIGNMENT != 0) {
        uint32_t scratch_cb_page_size = align(input_unit_size, DRAM_ALIGNMENT);
        tt_metal::CircularBufferConfig scratch_cb_out_config =
            tt_metal::CircularBufferConfig(1 * scratch_cb_page_size, {{scratch_cb_index, input_cb_data_format}})
                .set_page_size(scratch_cb_index, scratch_cb_page_size);
        auto cb_scratch = tt_metal::CreateCircularBuffer(program, all_cores, scratch_cb_out_config);
    }

    tt_metal::KernelHandle unary_reader_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)input_cb_index, (std::uint32_t)src_is_dram};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t)input_cb_index,
            (std::uint32_t)scratch_cb_index,
            (std::uint32_t)src_is_dram,
            (std::uint32_t)src_stick_size_is_power_of_two,
            (std::uint32_t)src_log2_stick_size};

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/"
            "reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    }

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::KernelHandle compute_kernel_id = 0;
    if (convert_df) {
        compute_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{});
    }

    uint32_t curr_idx_h = calculate_starting_idx_h(input, num_slices, slice_index);
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        uint32_t curr_num_units_per_shard = num_units_per_shard;
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core == end_core) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            curr_num_units_per_shard = shard_height * shard_width;
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(),
                 shard_height,
                 shard_width,
                 num_units_offset,
                 curr_num_units_per_shard,
                 curr_idx_h + curr_idx_w});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = input_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                    curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                        curr_num_units_per_shard = shard_height * num_units_per_shard_width;
                    }
                }
            }
            uint32_t padded_shard_width = align(shard_width, ADDRESS_ALIGNMENT);
            bool aligned = src_is_dram ? curr_idx_w % DRAM_ALIGNMENT == 0 : true;
            uint32_t aligned_width_offset, aligned_shard_width, aligned_offset;
            if (!aligned) {
                aligned_width_offset = round_down(curr_idx_w, DRAM_ALIGNMENT);
                aligned_offset = curr_idx_w - aligned_width_offset;
                aligned_shard_width = aligned_offset + shard_width;
            } else {
                aligned_width_offset = curr_idx_w;
                aligned_shard_width = shard_width;
                aligned_offset = 0;
            }

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core,
                {src_buffer->address(), num_units_per_row, shard_height, shard_width, padded_shard_width, static_cast<uint32_t>(aligned), aligned_width_offset, aligned_shard_width, aligned_offset, curr_idx_h});
            curr_idx_w += input_unit_size;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {curr_num_units_per_shard});
        if (convert_df) {
            tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {curr_num_units_per_shard});
        }
    }

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_output, cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        auto shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.grid;

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor& input, const Tensor& output, uint32_t num_slices, uint32_t slice_index) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, input_unit_size, output_unit_size, num_units_per_shard_width,
        num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last,
        num_units_per_shard_width_last;

    tt_metal::Device* device = input.device();

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout;

    bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    CoreCoord end_core = (*shard_spec.grid.ranges().rbegin()).end;
    if (output.get_layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        input_unit_size = tt_metal::detail::TileSize(input_cb_data_format);
        output_unit_size = tt_metal::detail::TileSize(output_cb_data_format);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = output.volume() / output.get_legacy_shape()[-1] / TILE_HEIGHT / num_slices;
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (output.volume() / output.get_legacy_shape()[-1] / shard_spec.shape[0]) *
                    (input.get_legacy_shape()[-1] / shard_spec.shape[1]);
        input_unit_size = shard_spec.shape[1] * input.element_size();
        output_unit_size = shard_spec.shape[1] * output.element_size();
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.get_legacy_shape()[-1] * output.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.get_legacy_shape()[-1];
        num_units_per_shard_height_last =
            num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            output_unit_size - (round_up(num_units_per_row, output_unit_size) - num_units_per_row);
    }

    bool convert_df = input_cb_data_format != output_cb_data_format;

    auto& all_cores = shard_spec.grid;
    uint32_t num_cores = all_cores.num_cores();

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t out_cb_index = src0_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t input_page_size = align(input_unit_size, ADDRESS_ALIGNMENT);
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_units * input_page_size, {{src0_cb_index, input_cb_data_format}})
            .set_page_size(src0_cb_index, input_page_size)
            .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);
    if (convert_df) {
        out_cb_index = CB::c_out0;
        uint32_t output_page_size = align(output_unit_size, ADDRESS_ALIGNMENT);
        tt_metal::CircularBufferConfig output_cb_out_config =
            tt_metal::CircularBufferConfig(num_input_units * output_page_size, {{out_cb_index, output_cb_data_format}})
                .set_page_size(out_cb_index, output_page_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_out_config);
    }

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_cb_index};

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    tt_metal::KernelHandle unary_writer_kernel_id;
    if (input.get_layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index, (std::uint32_t)dst_is_dram};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    } else {
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t)out_cb_index,
            (std::uint32_t)dst_is_dram,
            (std::uint32_t)dst_stick_size_is_power_of_two,
            (std::uint32_t)dst_log2_stick_size};

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/"
            "writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }
    if (convert_df) {
        vector<uint32_t> compute_kernel_args = {num_units_per_shard};

        auto eltwise_unary_kernel_group_1 = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/kernels/compute/eltwise_copy.cpp",
            all_cores,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args});
    }

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, all_cores, {num_units_per_shard});

    uint32_t curr_idx_h = calculate_starting_idx_h(output, num_slices, slice_index);
    uint32_t curr_idx_w = 0;

    const auto cores = corerange_to_cores(all_cores, std::nullopt, rm_orientation);
    for (const auto& core : cores) {
        if (input.get_layout() == Layout::TILE) {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = num_units_per_shard_width;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(),
                 num_units_per_shard_height,
                 num_units_per_shard_width,
                 shard_height,
                 shard_width,
                 num_units_offset,
                 num_units_per_shard,
                 curr_idx_h + curr_idx_w});
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = output_unit_size;
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
                if (core.x == end_core.x && core.y == end_core.y) {
                    shard_width = num_units_per_shard_width_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::BLOCK_SHARDED) {
                if (rm_orientation) {
                    if (core.x == end_core.x) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.y == end_core.y) {
                        shard_height = num_units_per_shard_height_last;
                    }
                } else {
                    if (core.y == end_core.y) {
                        shard_width = num_units_per_shard_width_last;
                    }
                    if (core.x == end_core.x) {
                        shard_height = num_units_per_shard_height_last;
                    }
                }
            }
            uint32_t padded_shard_width = align(shard_width, ADDRESS_ALIGNMENT);
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel_id,
                core,
                {dst_buffer->address(), num_units_per_row, shard_height, shard_width, padded_shard_width, curr_idx_w, curr_idx_h});
            curr_idx_w += output_unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }
    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cores, num_slices](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();

        Buffer* dst_buffer = nullptr;
        if (num_slices > 1) {
            // If we have num_slices > 1, it means that our op is S->I partial.
            // And currently we store output tensors there as input[1]
            dst_buffer = input_tensors.at(1).buffer();
        } else {
            dst_buffer = output_tensors.at(0).buffer();
        }

        auto shard_spec = input_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.grid;

        for (const auto& core : cores) {
            {
                auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

std::unordered_map<CoreCoord, std::vector<CorePageRange>> get_core_page_ranges(
    Buffer* input_buffer, Buffer* output_buffer) {
    const auto& output_shard_to_host_mapping = output_buffer->get_dev_page_to_host_page_mapping();
    const auto& input_page_to_local_page_mapping = input_buffer->get_host_page_to_local_shard_page_mapping();
    const auto& host_page_to_input_page_mapping = input_buffer->get_host_page_to_dev_page_mapping();

    auto num_pages = std::min<uint32_t>(output_shard_to_host_mapping.size(), input_buffer->num_pages());

    // First get output_core to vector< pair<input_core, input_page> (num_pages_in_output)
    std::unordered_map<CoreCoord, std::vector<std::pair<CoreCoord, uint32_t>>> output_core_to_vector_input_core_page;

    for (uint32_t output_page_id = 0; output_page_id < num_pages; output_page_id++) {
        auto output_core = output_buffer->get_core_from_dev_page_id(output_page_id);
        auto host_page = output_shard_to_host_mapping[output_page_id];
        auto input_page = host_page_to_input_page_mapping[host_page];
        auto local_input_page = input_page_to_local_page_mapping[host_page];
        auto input_core = input_buffer->get_core_from_dev_page_id(input_page);
        if (output_core_to_vector_input_core_page.find(output_core) == output_core_to_vector_input_core_page.end()) {
            output_core_to_vector_input_core_page[output_core] = {{input_core, local_input_page}};
        } else {
            output_core_to_vector_input_core_page[output_core].push_back({input_core, local_input_page});
        }
    }

    // now compress to output_core to vector<pair<input_core, input_page_range> (num_page_ranges_in_output)
    auto output_cores = corerange_to_cores(output_buffer->shard_spec().grid());
    std::unordered_map<CoreCoord, std::vector<CorePageRange>> ret_map;
    ret_map.reserve(output_cores.size());

    for (auto output_core : output_cores) {
        ret_map.try_emplace(output_core, std::vector<CorePageRange>{});
        const auto& input_cores_with_pages = output_core_to_vector_input_core_page.at(output_core);
        auto it = input_cores_with_pages.begin();
        const auto end = input_cores_with_pages.end();
        while (it != end) {
            const auto start_core = it->first;
            const auto start_page = it->second;
            auto expected_next_page = start_page + 1;

            // Find the end of the current range
            auto next_it = std::find_if(it + 1, end, [start_core, &expected_next_page](const auto& core_page) {
                const auto& [core, page] = core_page;
                bool is_same_range = (core == start_core) && (page == expected_next_page);
                expected_next_page += is_same_range;
                return !is_same_range;
            });

            ret_map[output_core].push_back(CorePageRange{start_core, {start_page, expected_next_page}});
            it = next_it;
        }
    }

    return ret_map;
}

operation::ProgramWithCallbacks reshard_multi_core(const Tensor& input, Tensor& output) {
    auto device = input.device();
    auto output_core_to_page_range_pair = get_core_page_ranges(input.buffer(), output.buffer());

    tt_metal::Program program{};

    auto input_shard_spec = input.shard_spec().value();
    auto output_shard_spec = output.shard_spec().value();
    auto all_cores = output_shard_spec.grid;

    uint32_t dst_cb_index = 16;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reshard_reader.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig({dst_cb_index}));

    std::vector<uint32_t> writer_compile_time_args = {dst_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto cores = corerange_to_cores(all_cores);

    uint32_t total_size, page_size, unit_size;
    auto output_shard_shape = output_shard_spec.shape;
    auto data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    if (input.get_layout() == Layout::TILE) {
        page_size = tt_metal::detail::TileSize(data_format);
        unit_size = page_size;
        total_size = output_shard_spec.numel() / TILE_HW * unit_size;
    } else {
        unit_size = output_shard_spec.shape[1] * output.element_size();
        page_size = output.get_legacy_shape()[-1] * output.element_size();
        total_size = output_shard_shape[0] * unit_size;
    }

    tt_metal::CircularBufferConfig cb_dst_config =
        tt_metal::CircularBufferConfig(
            total_size, {{dst_cb_index, data_format}})
            .set_page_size(dst_cb_index, unit_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_dst0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);

    for (auto core : cores) {
        auto page_range_vector = output_core_to_page_range_pair.at(core);
        uint32_t num_ranges = page_range_vector.size();
        std::vector<uint32_t> runtime_args = {input.buffer()->address(), 0, num_ranges};
        uint32_t num_output_pages = 0;
        for (const auto& [core, range] : page_range_vector) {
            auto physical_input_core = device->worker_core_from_logical_core(core);
            runtime_args.push_back(physical_input_core.x);
            runtime_args.push_back(physical_input_core.y);
            runtime_args.push_back(range.start * page_size);                // start addr_offset
            runtime_args.push_back((range.end - range.start) * page_size);  // size
            num_output_pages += range.end - range.start;
        }
        runtime_args[1] = num_output_pages;
        tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
    }

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id, page_size, cb_dst0](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        auto output_core_to_page_range_pair = get_core_page_ranges(src_buffer, dst_buffer);

        auto input_shard_spec = input_tensors.at(0).shard_spec().value();
        auto output_shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = input_shard_spec.grid.merge(output_shard_spec.grid);
        auto cores = corerange_to_cores(all_cores);
        auto device = input_tensors.at(0).device();

        for (const auto& core : cores) {
            auto page_range_vector = output_core_to_page_range_pair.at(core);
            uint32_t num_ranges = page_range_vector.size();
            std::vector<uint32_t> runtime_args = {src_buffer->address(), 0, num_ranges};
            uint32_t num_output_pages = 0;
            for (const auto& [core, range] : page_range_vector) {
                auto physical_input_core = device->worker_core_from_logical_core(core);
                runtime_args.push_back(physical_input_core.x);
                runtime_args.push_back(physical_input_core.y);
                runtime_args.push_back(range.start * page_size);
                runtime_args.push_back((range.end - range.start) * page_size);
                num_output_pages += range.end - range.start;
            }
            runtime_args[1] = num_output_pages;
            tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
            tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {num_output_pages});
            UpdateDynamicCircularBufferAddress(program, cb_dst0, *dst_buffer);
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
