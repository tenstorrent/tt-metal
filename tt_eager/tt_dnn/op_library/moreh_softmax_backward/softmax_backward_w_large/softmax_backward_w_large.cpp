// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.hpp"
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

operation::ProgramWithCallbacks moreh_softmax_backward_w_large(const Tensor &output, const Tensor &output_grad, const Tensor &input_grad, const CoreRange core_range, const MorehSoftmaxBackwardOp op) {
    log_info(LogTest, "Large tensor algorithm selected");
    // split work
    auto shape = input_grad.get_legacy_shape();
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    uint32_t num_kernel_rows = N * C * Ht;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, num_kernel_rows);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input_grad.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // output
            {CB::c_in1, 2},        // output_grad
            {CB::c_in2, 1},        // scaler
            {CB::c_in3, 1},        // mask
            {CB::c_out0, 2},       // input_grad
            {CB::c_intermed0, 1},  // output * output_grad
            {CB::c_intermed1, 1},  // reduce
            {CB::c_intermed2, 1},  // dy - sum
            {CB::c_intermed3, 2},  // add(output * output_grad)
        });

    // create read/wrtie kernel
    bool y_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dy_is_dram = output_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dx_is_dram = input_grad.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    std::map<string, string> compute_defines;
    if (op == MorehSoftmaxBackwardOp::SOFTMAX) compute_defines["SOFTMAX"] = "1";
    else compute_defines["SOFTMIN"] = "1";

    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        compute_defines["LOG"] = 1;
        reader_defines["LOG"] = 1;
    }

    auto reader_kernel_id = CreateReadKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_softmax_backward/kernels/reader_moreh_softmax_backward_w_large.cpp", all_cores, {y_is_dram, dy_is_dram}, reader_defines);
    auto writer_kernel_id = CreateWriteKernel(
        program, "tt_eager/tt_dnn/op_library/moreh_softmax_backward/kernels/writer_moreh_softmax_w.cpp", all_cores, {dx_is_dram}, writer_defines);


    // create compute kernel
    CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_softmax_backward/kernels/moreh_softmax_backward_w_large.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1, Wt}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2, Wt}},
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
        uint32_t mask_w = shape.without_padding()[3] % TILE_WIDTH;
        if(mask_w == 0) mask_w = TILE_WIDTH;
        vector<uint32_t> reader_args = {
            output.buffer()->address(),
            output_grad.buffer()->address(),
            num_tiles_per_core, tile_offset, Wt, *reinterpret_cast<uint32_t *>(&scaler), mask_w};

        vector<uint32_t> writer_args = {input_grad.buffer()->address(), num_tiles_per_core, tile_offset, Wt};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core * Wt;
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
        TT_ASSERT(input_buffers.size() == 2);
        TT_ASSERT(output_buffers.size() == 1);

        auto output_dram_buffer = input_buffers.at(0);
        auto output_grad_dram_buffer = input_buffers.at(1);
        auto input_grad_dram_buffer = output_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = output_dram_buffer->address();
                runtime_args[1] = output_grad_dram_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_grad_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
