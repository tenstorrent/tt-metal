// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

operation::ProgramWithCallbacks moreh_sgd_(
    const Tensor& param_in,
    const Tensor& grad,
    std::optional<const Tensor> momentum_buffer_in,
    const Tensor& param_out,
    std::optional<const Tensor> momentum_buffer_out,
    float lr,
    float momentum,
    float dampening,
    float weight_decay,
    bool nesterov,
    bool momentum_initialized,
    const CoreRange core_range) {
    // split work
    auto shape = param_in.get_legacy_shape();
    auto N = shape[0];
    auto C = shape[1];
    auto H = shape[2];
    auto W = shape[3];
    auto Ht = H / TILE_HEIGHT;
    auto Wt = W / TILE_WIDTH;

    bool has_momentum_buffer = momentum_buffer_in.has_value() && momentum_buffer_out.has_value();

    uint32_t units_to_divide = N * C * Ht * Wt;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(param_in.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_in0, 2},        // param_in
            {CB::c_in1, 2},        // grad
            {CB::c_in2, 2},        // momentum_in
            {CB::c_out0, 2},       // param_out
            {CB::c_out1, 2},       // momentum_out
            {CB::c_intermed0, 5},  // cb_scalar_args (lr, momentum, dampening, weight_decay, one)
            {CB::c_intermed1, 1},  //
            {CB::c_intermed2, 1},  //
            {CB::c_intermed3, 1},  //
            {CB::c_intermed4, 1},  //
        });

    std::map<string, string> reader_defines;
    std::map<string, string> writer_defines;
    std::map<string, string> compute_defines;

    if (weight_decay != 0) {
        reader_defines["WEIGHT_DECAY"] = 1;
        compute_defines["WEIGHT_DECAY"] = 1;
    }

    if (momentum != 0) {
        reader_defines["MOMENTUM"] = 1;
        compute_defines["MOMENTUM"] = 1;
        writer_defines["MOMENTUM"] = 1;
    }

    if (momentum_initialized) {
        reader_defines["MOMENTUM_INITIALIZED"] = 1;
        compute_defines["MOMENTUM_INITIALIZED"] = 1;
    }

    if (nesterov) {
        reader_defines["NESTEROV"] = 1;
        compute_defines["NESTEROV"] = 1;
    }

    // create read/wrtie kernel
    auto reader_kernel_ids = CreateReadKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_sgd/kernels/reader_moreh_sgd.cpp",
        all_cores,
        {
            is_dram(param_in),
            is_dram(grad),
            is_dram(momentum_buffer_in),
        },
        reader_defines);
    auto writer_kernel_ids = CreateWriteKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_sgd/kernels/writer_moreh_sgd.cpp",
        all_cores,
        {is_dram(param_out), is_dram(momentum_buffer_out)},
        writer_defines);

    // create compute kernel
    CreateComputeKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_sgd/kernels/moreh_sgd.cpp",
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
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
            TT_FATAL(false, "Core not in specified core ranges");
        }

        union { float f; uint32_t u; } u_lr, u_momentum, u_dampening, u_weight_decay, u_one;
        u_lr.f = lr;
        u_momentum.f = momentum;
        u_dampening.f = dampening;
        u_weight_decay.f = weight_decay;
        u_one.f = 1.0f;


        vector<uint32_t> reader_args = {
            param_in.buffer()->address(),
            grad.buffer()->address(),
            momentum_buffer_in.has_value() ? momentum_buffer_in.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
            u_lr.u,
            u_momentum.u,
            u_dampening.u,
            u_weight_decay.u,
            u_one.u,
            };

        vector<uint32_t> writer_args = {
            param_out.buffer()->address(),
            momentum_buffer_out.has_value() ? momentum_buffer_out.value().buffer()->address() : 0,
            num_tiles_per_core,
            tile_offset,
        };

        SetRuntimeArgs(program, reader_kernel_ids, core, reader_args);
        SetRuntimeArgs(program, writer_kernel_ids, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [reader_kernel_ids = reader_kernel_ids,
                                           writer_kernel_ids = writer_kernel_ids,
                                           num_cores,
                                           core_h,
                                           has_momentum_buffer = has_momentum_buffer](
        const void* operation,
        Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        const std::vector<Tensor>& output_tensors
        ) {
        TT_ASSERT(input_tensors.size() == 3);
        TT_ASSERT(optional_input_tensors.size() == 0 || optional_input_tensors.size() == 2);

        auto param_in_address = input_tensors.at(0).buffer()->address();
        auto grad_address = input_tensors.at(1).buffer()->address();
        auto param_out_address = input_tensors.at(2).buffer()->address();

        for (uint32_t core_i = 0; core_i < num_cores; core_i++) {
            CoreCoord core = {core_i / core_h, core_i % core_h};

            {
                auto& runtime_args = GetRuntimeArgs(program, reader_kernel_ids, core);
                runtime_args[0] = param_in_address;
                runtime_args[1] = grad_address;
                runtime_args[2] = param_out_address;

                if (has_momentum_buffer) {
                    auto momentum_buffer_in = optional_input_tensors.at(0).value().buffer();
                    TT_ASSERT(momentum_buffer_in != nullptr);
                    runtime_args[3] = momentum_buffer_in->address();
                }
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_ids, core);
                runtime_args[0] = param_out_address;

                if (has_momentum_buffer) {
                    auto momentum_buffer_out = optional_input_tensors.at(1).value().buffer();
                    TT_ASSERT(momentum_buffer_out != nullptr);
                    runtime_args[1] = momentum_buffer_out->address();
                }
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
