// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_binary_multi_core(const Tensor &a, const Tensor &b, const Tensor& output, BinaryOpType op_type, const std::optional<std::vector<UnaryWithParam>> fused_activations) {

    Program program{};

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.get_dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    tt_metal::Device *device = a.device();

    std::optional<ShardSpec> shard_spec = std::nullopt;
    bool src0_sharded = a.memory_config().is_sharded();
    bool src1_sharded = b.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool block_sharded = false;

    if (src0_sharded) {
        shard_spec = a.shard_spec().value();
        block_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (src1_sharded) {
        shard_spec = b.shard_spec().value();
        block_sharded = b.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    } else if (out_sharded) {
        shard_spec = output.shard_spec().value();
        block_sharded = output.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
    }

    uint32_t max_block_size = 1, num_tiles_per_shard = 0;
    if (shard_spec.has_value()) {
        num_tiles_per_shard = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
        max_block_size = find_max_block_size(num_tiles_per_shard);
    }

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = src0_sharded ? num_tiles_per_shard : 2 * max_block_size;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    if (src0_sharded) {
        cb_src0_config = cb_src0_config.set_globally_allocated_address(*a.buffer());
    }
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    num_input_tiles = src1_sharded ? num_tiles_per_shard : 2 * max_block_size;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
		.set_page_size(src1_cb_index, src1_single_tile_size);
    if (src1_sharded) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*b.buffer());
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_src1_config);

    std::map<string, string> eltwise_defines = eltwise_binary_op_utils::get_defines(op_type, fused_activations);

    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN0_0") != eltwise_defines.end()) {
        tt_metal::CircularBufferConfig cb_interm_config = tt_metal::CircularBufferConfig(1 * src0_single_tile_size, {{CB::c_intermed0, src0_cb_data_format}})
		    .set_page_size(CB::c_intermed0, src0_single_tile_size);
        auto cb_interm = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm_config);
    }
    if (eltwise_defines.find("SFPU_OP_INIT_PRE_IN1_0") != eltwise_defines.end()) {
        tt_metal::CircularBufferConfig cb_interm2_config = tt_metal::CircularBufferConfig(1 * src1_single_tile_size, {{CB::c_intermed1, src1_cb_data_format}})
		    .set_page_size(CB::c_intermed1, src1_single_tile_size);
        auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_interm2_config);
    }

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = (out_sharded || block_sharded) ? num_tiles_per_shard : 2 * max_block_size;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
        .set_page_size(output_cb_index, dst_single_tile_size);
    if (out_sharded) {
        cb_output_config = cb_output_config.set_globally_allocated_address(*output.buffer());
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, cb_output_config);

    std::map<string, string> reader_defines;
    if (src0_sharded) {
        reader_defines["IN0_SHARDED"] = "1";
    }
    if (src1_sharded) {
        reader_defines["IN1_SHARDED"] = "1";
    }
    std::map<string, string> writer_defines;
    if (out_sharded) {
        writer_defines["OUT_SHARDED"] = "1";
    }

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) src1_is_dram
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/eltwise_binary/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        (block_sharded and not out_sharded) ? "tt_eager/tt_dnn/op_library/sharded/kernels/dataflow/writer_unary_sharded_blocks_interleaved_start_id.cpp" : "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_device_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_binary.cpp",
        all_device_cores,
        tt_metal::ComputeConfig{.defines = eltwise_defines}
    );

    auto set_runtime_args = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
            eltwise_binary_kernel_id,
            cb_src0,
            cb_src1,
            cb_output,
            compute_with_storage_grid_size,
            src0_single_tile_size,
            src1_single_tile_size,
            dst_single_tile_size
        ]
    (
        Program& program,
        const Tensor& a,
        const Tensor& b,
        const Tensor& output
    ) {
        auto src_buffer_a = a.buffer();
        auto src_buffer_b = b.buffer();
        auto dst_buffer = output.buffer();

        CoreRangeSet all_cores({}), core_group_1({}), core_group_2({});

        std::optional<ShardSpec> shard_spec = std::nullopt;
        bool src0_sharded = a.memory_config().is_sharded();
        bool src1_sharded = b.memory_config().is_sharded();
        bool out_sharded = output.memory_config().is_sharded();

        bool block_sharded = false;
        if (src0_sharded) {
            shard_spec = a.shard_spec().value();
            block_sharded = a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        } else if (src1_sharded) {
            shard_spec = b.shard_spec().value();
            block_sharded = b.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        } else if (out_sharded) {
            shard_spec = output.shard_spec().value();
            block_sharded = output.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED;
        }

        uint32_t num_tiles = a.volume() / TILE_HW;

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        uint32_t num_cores, num_tiles_per_core_group_1, num_tiles_per_core_group_2;

        uint32_t block_size_per_core_group_1 = 1, block_size_per_core_group_2 = 1, max_block_size = 1;

        uint32_t block_cnt_per_core_group_1, block_cnt_per_core_group_2;

        bool row_major;
        uint32_t block_height = 0, block_width = 0, block_size = 0, output_width = 0, last_unpadded_block_height = 0, last_unpadded_block_width = 0;
        CoreCoord end_core;
        vector<CoreCoord> cores;

        if (shard_spec.has_value()) {
            all_cores = shard_spec.value().grid;
            num_cores = all_cores.num_cores();
            core_group_1 = all_cores;
            core_group_2 = CoreRangeSet({});
            num_tiles_per_core_group_1 = shard_spec.value().shape[0] * shard_spec.value().shape[1] / TILE_HW;
            num_tiles_per_core_group_2 = 0;
            block_size_per_core_group_1 = find_max_block_size(num_tiles_per_core_group_1);
            max_block_size = block_size_per_core_group_1;

            block_cnt_per_core_group_1 = num_tiles_per_core_group_1 / block_size_per_core_group_1;
            block_cnt_per_core_group_2 = num_tiles_per_core_group_2 / block_size_per_core_group_2;
            row_major = shard_spec.value().orientation == ShardOrientation::ROW_MAJOR;
            if (block_sharded) {
                block_height = shard_spec.value().shape[0] / TILE_HEIGHT;
                block_width = shard_spec.value().shape[1] / TILE_WIDTH;
                block_size = block_width * block_height;
                end_core = (*shard_spec.value().grid.ranges().begin()).end;
                output_width = output.get_legacy_shape()[-1] / TILE_WIDTH;
                uint32_t output_height = output.volume() / output.get_legacy_shape()[-1] / TILE_HEIGHT;
                last_unpadded_block_height = block_height -  (round_up(output_height, block_height) - output_height);
                last_unpadded_block_width = block_width -  (round_up(output_width, block_width) - output_width);
            }
            auto bbox = core_group_1.bounding_box();
            cores = grid_to_cores_with_noop(bbox.end.x, bbox.end.y, num_cores_x, num_cores_y, row_major);
        } else {
            row_major = true;
            std::tie(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_tiles, row_major);
            block_cnt_per_core_group_1 = num_tiles_per_core_group_1;
            block_cnt_per_core_group_2 = num_tiles_per_core_group_2;
            cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, row_major);
        }

        uint32_t g1_numcores = core_group_1.num_cores();
        uint32_t g2_numcores = core_group_2.num_cores();

        std::vector< std::vector<uint32_t> > binary_reader_args = { cores.size(), std::vector<uint32_t>(4)  };
        std::vector< std::vector<uint32_t> > eltwise_binary_args = { cores.size(), std::vector<uint32_t>(2)  };
        std::vector< std::vector<uint32_t> > unary_writer_args;

        if (block_sharded and not out_sharded)
            unary_writer_args = { cores.size(), std::vector<uint32_t>(7) };
        else
            unary_writer_args = { cores.size(), std::vector<uint32_t>(3) };

        for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; ++i){
            const CoreCoord &core = cores.at(i);
            uint32_t num_tiles_per_core = 0;
            uint32_t block_cnt_per_core = 0;
            uint32_t block_size_per_core = 0;
            if (i < g1_numcores) {
                num_tiles_per_core = num_tiles_per_core_group_1;
                block_cnt_per_core = block_cnt_per_core_group_1;
                block_size_per_core = block_size_per_core_group_1;
            } else {
                num_tiles_per_core = num_tiles_per_core_group_2;
                block_cnt_per_core = block_cnt_per_core_group_2;
                block_size_per_core = block_size_per_core_group_2;
            }
            binary_reader_args[i] =  {src_buffer_a->address(), src_buffer_b->address(), num_tiles_per_core, num_tiles_read};
            eltwise_binary_args[i] = {block_cnt_per_core, block_size_per_core};
            if (block_sharded and not out_sharded) {
                uint32_t block_start_width_offset;
                uint32_t block_start_height_offset;
                uint32_t unpadded_block_height = block_height;
                uint32_t unpadded_block_width = block_width;
                if (row_major) {
                    block_start_width_offset = core.x * block_width;
                    block_start_height_offset = core.y * block_height;
                    if (core.x == end_core.x) {
                        unpadded_block_width = last_unpadded_block_width;
                    }
                    if (core.y == end_core.y) {
                        unpadded_block_height = last_unpadded_block_height;
                    }
                } else {
                    block_start_width_offset = core.y * block_width;
                    block_start_height_offset = core.x * block_height;
                    if (core.y == end_core.y) {
                        unpadded_block_width = last_unpadded_block_width;
                    }
                    if (core.x == end_core.x) {
                        unpadded_block_height = last_unpadded_block_height;
                    }
                }
                unary_writer_args[i] =  {   dst_buffer->address(),
                                            block_height,
                                            block_width,
                                            unpadded_block_height,
                                            unpadded_block_width,
                                            output_width,
                                            block_size,
                                            block_start_height_offset * output_width + block_start_width_offset };
            } else {
                unary_writer_args[i] = { dst_buffer->address(), num_tiles_per_core, num_tiles_read };
            }
            num_tiles_read += num_tiles_per_core;
        }

        SetRuntimeArgs(program, binary_reader_kernel_id, cores, binary_reader_args);
        SetRuntimeArgs(program, eltwise_binary_kernel_id, cores, eltwise_binary_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);

        if (src0_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer_a);
            UpdateCircularBufferTotalSize(program, cb_src0, num_tiles_per_core_group_1 * src0_single_tile_size);
        }
        if (src1_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_src1, *src_buffer_b);
            UpdateCircularBufferTotalSize(program, cb_src1, num_tiles_per_core_group_1 * src1_single_tile_size);
        }
        if (out_sharded) {
            UpdateDynamicCircularBufferAddress(program, cb_output, *dst_buffer);
            UpdateCircularBufferTotalSize(program, cb_output, num_tiles_per_core_group_1 * dst_single_tile_size);
        }
    };

    set_runtime_args(program, a, b, output);

    auto override_runtime_arguments_callback = [
            set_runtime_args
        ]
    (
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();
        const auto& output_tensor = output_tensors.size() == 1 ? output_tensors.at(0) : input_tensors.at(0);

        set_runtime_args(program, input_tensors.at(0), input_tensors.at(1), output_tensor);
    };
    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
