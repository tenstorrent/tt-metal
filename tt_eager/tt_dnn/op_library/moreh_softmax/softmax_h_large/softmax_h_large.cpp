// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.hpp"
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

operation::ProgramWithCallbacks moreh_softmax_h_large(const Tensor &input, const Tensor &output, const CoreRange core_range, const MorehSoftmaxOp op) {
    log_info(LogTest, "Large tensor algorithm selected");
    // split work
    auto shape = input.get_legacy_shape();
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    uint32_t num_cols_tiles = N * C * Wt;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, num_cols_tiles);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // input
            {CB::c_in1, 1},        // mask
            {CB::c_out0, 2},       // output
            {CB::c_intermed0, 2},  // exp(x)
            {CB::c_intermed1, 1},  // reduce
            {CB::c_intermed2, 1},  // sum
            {CB::c_in2, 1}         // scaler
        });

    // create read/wrtie kernel
    bool src_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;

    auto reader_kernel_id = CreateReadKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_softmax/kernels/reader_moreh_softmax_h_large.cpp", all_cores, {src_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_softmax/kernels/writer_moreh_softmax_h.cpp", all_cores, {dst_is_dram}, writer_defines);

    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxOp::SOFTMAX || op == MorehSoftmaxOp::LOGSOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["LOG"] = "1";
    }

    // create compute kernel
    CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_softmax/kernels/moreh_softmax_h_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Ht}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Ht}},
        },
        compute_defines);

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

        float scaler = 1.0f;
        uint32_t mask_h = shape.without_padding()[2] % TILE_HEIGHT;
        if(mask_h == 0) mask_h = TILE_HEIGHT;
        vector<uint32_t> reader_args = {
            input.buffer()->address(), num_tiles_per_core, tile_offset, Ht, Wt, *reinterpret_cast<uint32_t *>(&scaler), mask_h};

        vector<uint32_t> writer_args = {output.buffer()->address(), num_tiles_per_core, tile_offset, Ht, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

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
