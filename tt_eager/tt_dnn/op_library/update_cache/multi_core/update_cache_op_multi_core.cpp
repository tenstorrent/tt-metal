// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/update_cache/update_cache_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks update_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t update_idx) {
    Program program{};

    tt::DataFormat cache_cb_data_format = tt_metal::datatype_to_dataformat_converter(cache_tensor.dtype());
    uint32_t cache_single_tile_size = tt_metal::detail::TileSize(cache_cb_data_format);

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    tt::DataFormat interm_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t interm_single_tile_size = tt_metal::detail::TileSize(interm_cb_data_format);

    uint32_t Wt = cache_tensor.shape()[-1] / TILE_WIDTH;

    // Width size after untilize
    uint32_t Wbytes = cache_tensor.shape()[-1] * sizeof(bfloat16);

    uint32_t cache_total_num_tiles = cache_tensor.volume() / TILE_HW;
    uint32_t cache_batch_num_tiles = cache_total_num_tiles / cache_tensor.shape()[0];
    uint32_t cache_head_num_tiles = cache_batch_num_tiles / cache_tensor.shape()[1];

    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    uint32_t B = input_tensor.shape()[-2];
    uint32_t num_batched_heads = input_tensor.shape()[1] * B / TILE_HEIGHT;
    uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
    tt_metal::Device *device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    bool row_major;
    uint32_t num_cores, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2;

    CoreRangeSet all_cores({}), core_group_1({}), core_group_2({});

    std::optional<ShardSpec> shard_spec = input_tensor.shard_spec();

    uint32_t num_input_tiles;
    if (shard_spec.has_value()) {
        row_major = shard_spec.value().shard_orientation == ShardOrientation::ROW_MAJOR;
        all_cores = shard_spec.value().shard_grid;
        num_cores = all_cores.num_cores();
        core_group_1 = all_cores;
        core_group_2 = CoreRangeSet({});
        num_batched_heads_per_core_group_1 = shard_spec.value().shard_shape[0] / TILE_HEIGHT;
        num_batched_heads_per_core_group_2 = 0;
        num_input_tiles = shard_spec.value().shard_shape[0] * shard_spec.value().shard_shape[1] / TILE_HW;
        auto bbox = all_cores.bounding_box();
        num_cores_x = bbox.end.x + 1;
        num_cores_y = bbox.end.y + 1;
    } else {
        row_major = true;
        std::tie(num_cores, all_cores, core_group_1, core_group_2, num_batched_heads_per_core_group_1, num_batched_heads_per_core_group_2) = split_work_to_cores(compute_with_storage_grid_size, num_batched_heads, row_major);
        num_input_tiles = 2 * Wt;
    }
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_cache_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_cache_tiles * cache_single_tile_size, {{src0_cb_index, cache_cb_data_format}})
		.set_page_size(src0_cb_index, cache_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * input_single_tile_size, {{src1_cb_index, input_cb_data_format}})
		.set_page_size(src1_cb_index, input_single_tile_size);
    if (shard_spec.has_value()) {
        cb_src1_config = cb_src1_config.set_globally_allocated_address(*input_tensor.buffer());
    }
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);
    uint32_t interm0_cb_index = CB::c_intermed0;
    uint32_t interm1_cb_index = CB::c_intermed1;
    uint32_t num_interm_tiles = Wt;
    std::map<uint8_t, tt::DataFormat> interim_data_format_spec = {
        {interm0_cb_index, interm_cb_data_format},
        {interm1_cb_index, interm_cb_data_format}
    };
    tt_metal::CircularBufferConfig cb_interm0_config = tt_metal::CircularBufferConfig(num_interm_tiles * interm_single_tile_size, interim_data_format_spec)
		.set_page_size(interm0_cb_index, interm_single_tile_size)
        .set_page_size(interm1_cb_index, interm_single_tile_size);
    auto cb_interm0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm0_config);

    uint32_t interm2_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig cb_interm2_config = tt_metal::CircularBufferConfig(num_interm_tiles * interm_single_tile_size, {{interm2_cb_index, interm_cb_data_format}})
		.set_page_size(interm2_cb_index, interm_single_tile_size);
    auto cb_interm2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_interm2_config);

    // Output is same tensor as cache input, so cb/tile size is same
    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = 2 * Wt;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * cache_single_tile_size, {{output_cb_index, cache_cb_data_format}})
		.set_page_size(output_cb_index, cache_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t)dst_is_dram,
        (uint32_t)src_is_dram,
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index
    };

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) output_cb_index,
        (std::uint32_t) interm0_cb_index,
        (std::uint32_t) interm1_cb_index,
        (std::uint32_t) interm2_cb_index
    };

    std::map<string, string> reader_kernel_defines;
    if (shard_spec.has_value()) {
        reader_kernel_defines["INPUT_SHARDED"] = "1";
    }

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/update_cache/kernels/dataflow/reader_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args, .defines = reader_kernel_defines});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/update_cache/kernels/dataflow/writer_update_cache_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {
        src0_cb_index,
        src1_cb_index,
        interm0_cb_index,
        interm1_cb_index,
        interm2_cb_index,
        output_cb_index,
        num_batched_heads_per_core_group_1,
        Wt
    };

    auto compute_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/update_cache/kernels/compute/update_cache.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );

    if (!core_group_2.ranges().empty()) {
        compute_kernel_args[6] = num_batched_heads_per_core_group_2;
        auto compute_kernel_group_2_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/update_cache/kernels/compute/update_cache.cpp",
        core_group_2,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
    );
    }

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    uint32_t cache_tile_idx = update_idx / TILE_HEIGHT * Wt;
    uint32_t cache_start_id = 0;
    uint32_t input_start_id = 0;
    uint32_t batch_start_id = 0;
    uint32_t total_batched_heads = 0;
    std::vector<uint32_t> cache_start_ids;
    cache_start_ids.reserve(num_cores);
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; ++i) {
        const CoreCoord &core = cores.at(i);
        uint32_t num_batched_heads_per_core;
        if (i < g1_numcores) {
            num_batched_heads_per_core = num_batched_heads_per_core_group_1;
        } else {
            num_batched_heads_per_core = num_batched_heads_per_core_group_2;
        }
        input_start_id = total_batched_heads * Wt;
        batch_start_id = (total_batched_heads * TILE_HEIGHT) % B;
        // Batch Offset + Head Offset + Index Offset
        cache_start_id = batch_start_id * cache_batch_num_tiles + ((total_batched_heads * TILE_HEIGHT) / B) * cache_head_num_tiles;
        cache_start_ids.push_back(cache_start_id);
        cache_start_id += cache_tile_idx;
        SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                dst_buffer->address(),
                src_buffer->address(),
                Wt, B, num_batched_heads_per_core, cache_total_num_tiles, cache_batch_num_tiles, cache_head_num_tiles, cache_start_id, input_start_id, batch_start_id
            }
        );

        SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                Wt, B, num_batched_heads_per_core, cache_total_num_tiles, cache_batch_num_tiles, cache_head_num_tiles, cache_start_id, batch_start_id, Wbytes, tile_update_offset
            }
        );
        total_batched_heads += num_batched_heads_per_core;
    }

    auto override_runtime_arguments_callback = [
        unary_reader_kernel_id,
        unary_writer_kernel_id,
        cores,
        Wbytes,
        Wt,
        cache_start_ids,
        cb_src0
    ](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto update_idx = static_cast<const UpdateCache*>(operation)->update_idx;

        uint32_t tile_update_offset = update_idx % TILE_HEIGHT * Wbytes;
        uint32_t cache_tile_idx = update_idx / TILE_HEIGHT * Wt;

        auto src_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = input_tensors.at(0).buffer();

        if (input_tensors.at(1).is_sharded()) {
            UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        }

        for (uint32_t i = 0, num_tiles_read = 0; i < cores.size(); ++i){
            const CoreCoord &core = cores.at(i);
            uint32_t curr_cache_start_id = cache_start_ids[i] + cache_tile_idx;
            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[1] = src_buffer->address();
                runtime_args[8] = curr_cache_start_id;
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[7] = curr_cache_start_id;
                runtime_args[10] = tile_update_offset;
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

operation::ProgramWithCallbacks fill_cache_multi_core(const Tensor& cache_tensor, const Tensor &input_tensor, const uint32_t batch_idx, const uint32_t update_idx) {
    Program program{};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);


    uint32_t num_tiles = input_tensor.volume() / TILE_HW;

    uint32_t cache_Ht = cache_tensor.shape()[-2] / TILE_HEIGHT, cache_Wt = cache_tensor.shape()[-1] / TILE_WIDTH;
    uint32_t cache_HtWt = cache_Ht * cache_Wt;
    uint32_t update_idxt = update_idx / TILE_HEIGHT;
    uint32_t start_idx = batch_idx * cache_HtWt + update_idxt * cache_Wt;
    tt_metal::Device *device = input_tensor.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = src0_cb_index;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = cache_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_tiles_per_core,
                num_tiles_written
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_tiles_per_core,
                start_idx + num_tiles_written
            }
        );
        num_tiles_written+=num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_y,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2,
            cache_HtWt,
            cache_Wt
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        const auto batch_idx = static_cast<const UpdateCache*>(operation)->batch_idx;
        const auto update_idx = static_cast<const UpdateCache*>(operation)->update_idx;

        uint32_t update_idxt = update_idx / TILE_HEIGHT;
        uint32_t start_idx = batch_idx * cache_HtWt + update_idxt * cache_Wt;

        auto src_buffer = input_tensors.at(1).buffer();

        auto dst_buffer = input_tensors.at(0).buffer();

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core = 0;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
                runtime_args[2] = start_idx + num_tiles_written;
            }
            num_tiles_written += num_tiles_per_core;
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
