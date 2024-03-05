// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_sum_backward_program(const Tensor &output_grad, const Tensor &input_grad) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto *device = output_grad.device();
    auto program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    const auto single_tile_size = detail::TileSize(cb_data_format);

    const auto &output_grad_shape = output_grad.get_legacy_shape();
    const auto &output_grad_shape_wo_padding = output_grad_shape.without_padding();
    const auto output_grad_n = output_grad_shape[0];
    const auto output_grad_c = output_grad_shape[1];
    const auto output_grad_ht = output_grad_shape[2] / TILE_HEIGHT;
    const auto output_grad_wt = output_grad_shape[3] / TILE_WIDTH;
    const auto output_grad_origin_h = output_grad_shape_wo_padding[2];
    const auto output_grad_origin_w = output_grad_shape_wo_padding[3];

    const auto &input_grad_shape = input_grad.get_legacy_shape();
    const auto &input_grad_shape_wo_padding = input_grad_shape.without_padding();
    const auto input_grad_n = input_grad_shape[0];
    const auto input_grad_c = input_grad_shape[1];
    const auto input_grad_ht = input_grad_shape[2] / TILE_HEIGHT;
    const auto input_grad_wt = input_grad_shape[3] / TILE_WIDTH;
    const auto input_grad_origin_h = input_grad_shape_wo_padding[2];
    const auto input_grad_origin_w = input_grad_shape_wo_padding[3];

    const auto n_need_bcast = (output_grad_n != input_grad_n);
    const auto c_need_bcast = (output_grad_c != input_grad_c);
    const auto ht_need_bcast = (output_grad_origin_h != input_grad_origin_h);
    const auto wt_need_bcast = (output_grad_origin_w != input_grad_origin_w);
    const auto num_input_grad_tiles = input_grad.volume() / TILE_HW;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {core_grid.x_, num_cores_y};

    const uint32_t in0_t = 2;   // input
    const uint32_t in1_t = 1;   // zero
    const uint32_t out0_t = 2;  // output
    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, num_input_grad_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // zero
            {CB::c_out0, out0_t},  // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    const auto reader_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/kernels/reader_moreh_sum_backward.cpp";
    const auto writer_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/kernels/writer_moreh_sum_backward.cpp";
    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> compute_args_group_1{num_cols_per_core_group_1};
    std::map<string, string> compute_defines;
    const auto compute_kernel_file = "tt_eager/tt_dnn/op_library/moreh_sum_backward/kernels/moreh_sum_backward.cpp";
    const auto compute_kernel_1_id = CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    std::optional<KernelHandle> compute_kernel_2_id = std::nullopt;
    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{num_cols_per_core_group_2};
        compute_kernel_2_id = CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {output_grad.buffer()->address(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(is_dram(output_grad)),
             output_grad_n,
             output_grad_c,
             output_grad_ht,
             output_grad_wt,
             input_grad_n,
             input_grad_c,
             input_grad_ht,
             input_grad_wt,
             n_need_bcast,
             c_need_bcast,
             ht_need_bcast,
             wt_need_bcast});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {input_grad.buffer()->address(),
             num_tiles_per_core,
             tile_offset,
             static_cast<uint32_t>(is_dram(input_grad))});

        if (core_group_1.core_coord_in_core_ranges(core)) {
            SetRuntimeArgs(
                program,
                compute_kernel_1_id,
                core,
                {num_tiles_per_core, n_need_bcast, c_need_bcast, ht_need_bcast, wt_need_bcast});
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            TT_ASSERT(compute_kernel_2_id.has_value());
            SetRuntimeArgs(
                program,
                compute_kernel_2_id.value(),
                core,
                {num_tiles_per_core, n_need_bcast, c_need_bcast, ht_need_bcast, wt_need_bcast});
        } else {
            TT_ASSERT(false, "Core not in specified core ranges.");
        }
        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id, num_cores_to_be_used, num_cores_y](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto *output_grad_buffer = input_tensors.at(0).buffer();
        const auto *input_grad_buffer = input_tensors.at(1).buffer();
        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = output_grad_buffer->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = input_grad_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
