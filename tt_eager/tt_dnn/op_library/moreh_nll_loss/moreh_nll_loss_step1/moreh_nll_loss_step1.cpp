// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_nll_loss_step1_impl(const Tensor &input, const Tensor &target, const std::optional<const Tensor> weight, Tensor &output, const int32_t ignore_index, const bool reduction_mean, const CoreRange core_range) {
    // split work
    auto input_shape = input.get_legacy_shape();
    auto N = input_shape[0];
    auto C = input_shape[1];
    auto H = input_shape[2];
    auto W = input_shape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    const auto input_shape_without_padding = input_shape.without_padding();
    const auto origin_N = input_shape_without_padding[0];
    const auto origin_C = input_shape_without_padding[1];
    const auto origin_H = input_shape_without_padding[2];
    const auto origin_W = input_shape_without_padding[3];

    const bool weight_has_value = weight.has_value();

    uint32_t num_tiles = N * Ht * Wt;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, num_tiles);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    tt::DataFormat target_cb_data_format = tt_metal::datatype_to_dataformat_converter(target.get_dtype());
    uint32_t single_tile_size = 2 * 1024;
    uint32_t target_cb_index = CB::c_in1;

    auto target_tile_bytes = tt_metal::detail::TileSize(target_cb_data_format);
    tt_metal::CircularBufferConfig cb_target_config = tt_metal::CircularBufferConfig(tt_metal::detail::TileSize(target_cb_data_format), {{target_cb_index, target_cb_data_format}})
        .set_page_size(target_cb_index, tt_metal::detail::TileSize(target_cb_data_format));
    auto cb_target = tt_metal::CreateCircularBuffer(program, all_cores, cb_target_config);

    uint32_t weight_num_tile = (C + single_tile_size - 1) / single_tile_size;
    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 1},         // input
            {CB::c_in2, static_cast<uint32_t>(weight_has_value ? weight_num_tile : 0)},         // weight
            {CB::c_in3, 1},         // one
            {CB::c_intermed0, 1},   // tmp_weight to reduce
            {CB::c_out0, 1},        // output
        });

    // create read/wrtie kernel
    const std::vector<uint32_t> reader_compile_time_args{
        target_tile_bytes,
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(target)),
        static_cast<uint32_t>(is_dram(weight)),
        static_cast<uint32_t>(weight_has_value)};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(output))};

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/reader_moreh_nll_loss_step1.cpp", all_cores, reader_compile_time_args, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/writer_moreh_nll_loss_step1.cpp", all_cores, writer_compile_time_args, writer_defines);

    // create compute kernel
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";

    const auto compute_kernel_ids = CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/kernels/moreh_nll_loss_step1_kernel.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
        },
        compute_defines);

    const auto input_addr = input.buffer()->address();
    const auto target_addr = target.buffer()->address();
    const auto weight_addr = weight_has_value ? weight.value().buffer()->address() : 0;
    const auto output_addr = output.buffer()->address();

    // Set Runtime Args
    auto core_x_offset = core_range.start.x;
    auto core_y_offset = core_range.start.y;
    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        vector<uint32_t> reader_args = {
            input_addr, target_addr, weight_addr, static_cast<uint32_t>(ignore_index),
            num_tiles_per_core, tile_offset, C, Ht * Wt, origin_H, origin_W};

        vector<uint32_t> writer_args = {output_addr, num_tiles_per_core, tile_offset};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_tiles_per_core, tile_offset};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[0], core, compute_runtime_args);
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(program, compute_kernel_ids[1], core, compute_runtime_args);
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_kernel_id,
            writer_kernel_id=writer_kernel_id,
            num_cores,
            core_h
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {
        TT_ASSERT(input_buffers.size() == 1);
        TT_ASSERT(output_buffers.size() == 1);

        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
