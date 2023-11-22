// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/sharded/sharded_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tensor/tensor_utils.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks interleaved_to_sharded_multi_core(const Tensor &input, Tensor &output, const CoreCoord& grid_size) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, unit_size, num_units_per_shard_width, num_units_per_shard_height, num_units_offset, num_units_per_row;

    tt_metal::Device *device = input.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto shard_spec = output.shard_spec().value();

    bool rm_orientation = shard_spec.shard_orientation == ShardOrientation::ROW_MAJOR;

    if (input.layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
        unit_size = tt_metal::detail::TileSize(cb_data_format);
        num_units_per_shard_height = shard_spec.shard_shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shard_shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row - num_units_per_shard_width;
    } else {
        num_units = (input.volume() / input.shape()[-1] / shard_spec.shard_shape[0]) * (input.shape()[-1] / shard_spec.shard_shape[1]);
        unit_size = shard_spec.shard_shape[1] * input.element_size();
        num_units_per_shard_height = shard_spec.shard_shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.shape()[-1] * input.element_size();
        num_units_offset = 1;
    }

    auto all_cores = shard_spec.shard_grid;
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;
    uint32_t out_cb_index = 0;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t page_size = round_up_to_mul32(unit_size);
    tt_metal::CircularBufferConfig cb_out_config = tt_metal::CircularBufferConfig(num_input_units * page_size, {{out_cb_index, cb_data_format}})
		.set_page_size(out_cb_index, page_size).set_globally_allocated_address(*output.buffer());
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::optional<KernelHandle> unary_reader_kernel_id;
    if (input.layout() == Layout::TILE) {

        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) out_cb_index,
            (std::uint32_t) src_is_dram
        };

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});
    } else {
        bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) out_cb_index,
            (std::uint32_t) src_is_dram,
            (std::uint32_t) src_stick_size_is_power_of_two,
            (std::uint32_t) src_log2_stick_size
        };

        unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});
    }

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index};
    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel_id,
        all_cores,
        {
            num_units_per_shard
        }
    );

    uint32_t curr_idx_h = 0, curr_idx_w = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = rm_orientation ? CoreCoord(i % num_cores_x, i / num_cores_x) : CoreCoord(i / num_cores_y, i % num_cores_y);

        if (!all_cores.core_coord_in_core_ranges(core)) {
            continue;
        }

        if (input.layout() == Layout::TILE) {
            tt_metal::SetRuntimeArgs(
                unary_reader_kernel_id.value(),
                core,
                {
                    src_buffer->address(),
                    num_units_per_shard_height,
                    num_units_per_shard_width,
                    num_units_offset,
                    num_units_per_shard,
                    curr_idx_h + curr_idx_w
                }
            );
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            tt_metal::SetRuntimeArgs(
                unary_reader_kernel_id.value(),
                core,
                {
                    src_buffer->address(),
                    num_units_per_row,
                    num_units_per_shard_height,
                    unit_size,
                    curr_idx_w,
                    curr_idx_h
                }
            );
            curr_idx_w += unit_size;
            if (curr_idx_w == num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }

    auto override_runtime_arguments_callback = [
            reader_kh=unary_reader_kernel_id.value(),
            unary_writer_kernel_id,
            cb_output,
            num_cores,
            num_cores_x,
            num_cores_y,
            rm_orientation
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        auto shard_spec = output_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.shard_grid;

        for (uint32_t i = 0; i < num_cores; ++i){
            CoreCoord core = rm_orientation ? CoreCoord(i % num_cores_x, i / num_cores_x) : CoreCoord(i / num_cores_y, i % num_cores_y);
            {
                if (!all_cores.core_coord_in_core_ranges(core)) {
                    continue;
                }
                auto &runtime_args = GetRuntimeArgs(reader_kh, core);
                runtime_args[0] = src_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress( cb_output, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks sharded_to_interleaved_multi_core(const Tensor &input, Tensor &output, const CoreCoord& grid_size) {
    tt_metal::Program program{};

    uint32_t num_units, num_units_per_shard, unit_size, num_units_per_shard_width, num_units_per_shard_height, num_units_offset, num_units_per_row, num_units_per_shard_height_last, num_units_per_shard_width_last;

    tt_metal::Device *device = input.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());

    auto shard_spec = input.shard_spec().value();
    auto shard_strategy = input.memory_config().memory_layout;

    bool rm_orientation = shard_spec.shard_orientation == ShardOrientation::ROW_MAJOR;
    CoreCoord end_core = (*shard_spec.shard_grid.ranges().begin()).end;
    if (output.layout() == Layout::TILE) {
        num_units = input.volume() / TILE_HW;
        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
        unit_size = tt_metal::detail::TileSize(cb_data_format);
        num_units_per_shard_height = shard_spec.shard_shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shard_shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = output.shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = output.volume() / output.shape()[-1] / TILE_HEIGHT;
        num_units_per_shard_height_last = num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last = num_units_per_shard_width - (round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
    } else {
        num_units = (output.volume() / output.shape()[-1] / shard_spec.shard_shape[0]) * (input.shape()[-1] / shard_spec.shard_shape[1]);
        unit_size = shard_spec.shard_shape[1] * input.element_size();
        num_units_per_shard_height = shard_spec.shard_shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row =output.shape()[-1] * output.element_size();
        num_units_offset = 1;
        uint32_t num_units_height = input.volume() / input.shape()[-1];
        num_units_per_shard_height_last = num_units_per_shard_height - (round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last = unit_size - (round_up(num_units_per_row, unit_size) - num_units_per_row);
    }

    auto all_cores = shard_spec.shard_grid;
    uint32_t num_cores_x = grid_size.x;
    uint32_t num_cores_y = grid_size.y;
    uint32_t num_cores = num_cores_x * num_cores_y;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t page_size = round_up_to_mul32(unit_size);
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_units * page_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, page_size).set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    auto src_buffer = input.buffer();

    auto dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});


    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::optional<tt_metal::KernelHandle> unary_writer_kernel_id;
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) dst_is_dram
        };

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});
    } else {
        bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(num_units_per_row);
        uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(num_units_per_row) : 0;
        std::vector<uint32_t> reader_compile_time_args = {
            (std::uint32_t) src0_cb_index,
            (std::uint32_t) dst_is_dram,
            (std::uint32_t) dst_stick_size_is_power_of_two,
            (std::uint32_t) dst_log2_stick_size
        };

        unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp",
            all_cores,
            tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args});
    }

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel_id,
        all_cores,
        {
            num_units_per_shard
        }
    );

    uint32_t curr_idx_h = 0, curr_idx_w = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = rm_orientation ? CoreCoord(i % num_cores_x, i / num_cores_x) : CoreCoord(i / num_cores_y, i % num_cores_y);

        if (!all_cores.core_coord_in_core_ranges(core)) {
            continue;
        }

        if (input.layout() == Layout::TILE) {
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
                unary_writer_kernel_id.value(),
                core,
                {
                    dst_buffer->address(),
                    num_units_per_shard_height,
                    num_units_per_shard_width,
                    shard_height,
                    shard_width,
                    num_units_offset,
                    num_units_per_shard,
                    curr_idx_h + curr_idx_w
                }
            );
            curr_idx_w += num_units_per_shard_width;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_row * num_units_per_shard_height;
            }
        } else {
            uint32_t shard_height = num_units_per_shard_height;
            uint32_t shard_width = unit_size;
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
                unary_writer_kernel_id.value(),
                core,
                {
                    dst_buffer->address(),
                    num_units_per_row,
                    shard_height,
                    shard_width,
                    curr_idx_w,
                    curr_idx_h
                }
            );
            curr_idx_w += unit_size;
            if (curr_idx_w >= num_units_per_row) {
                curr_idx_w = 0;
                curr_idx_h += num_units_per_shard_height;
            }
        }
    }
    auto override_runtime_arguments_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id=unary_writer_kernel_id.value(),
            cb_src0,
            num_cores,
            num_cores_x,
            num_cores_y,
            rm_orientation
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer = input_tensors.at(0).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        auto shard_spec = input_tensors.at(0).shard_spec().value();
        auto all_cores = shard_spec.shard_grid;

        for (uint32_t i = 0; i < num_cores; ++i){
            CoreCoord core = rm_orientation ? CoreCoord(i % num_cores_x, i / num_cores_x) : CoreCoord(i / num_cores_y, i % num_cores_y);
            {
                if (!all_cores.core_coord_in_core_ranges(core)) {
                    continue;
                }
                auto &runtime_args = GetRuntimeArgs(unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
        UpdateDynamicCircularBufferAddress( cb_src0, *src_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
