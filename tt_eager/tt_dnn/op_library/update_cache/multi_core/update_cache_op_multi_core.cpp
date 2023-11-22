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
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

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
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_tiles_per_core,
                num_tiles_written
            }
        );

        tt_metal::SetRuntimeArgs(
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
                auto &runtime_args = GetRuntimeArgs(unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(unary_writer_kernel_id, core);
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
