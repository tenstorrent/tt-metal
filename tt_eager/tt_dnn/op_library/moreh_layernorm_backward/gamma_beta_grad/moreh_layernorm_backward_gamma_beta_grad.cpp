// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_layernorm_backward_gamma_beta_grad_impl(
    const Tensor& output_grad,
    const Tensor& input,
    const Tensor& mean,
    const Tensor& rstd,
    uint32_t normalized_dims,
    const std::optional<std::reference_wrapper<const Tensor>> gamma_grad,
    const std::optional<std::reference_wrapper<const Tensor>> beta_grad) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    Device* device = output_grad.device();
    Program program = Program();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.get_legacy_shape();

    const bool is_lastdim_layernorm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto output_grad_shape_without_padding = output_grad_shape.without_padding();

    const auto origin_H = output_grad_shape_without_padding[2];
    const auto origin_W = output_grad_shape_without_padding[3];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && is_lastdim_layernorm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    auto adjusted_output_grad_shape = output_grad_shape;
    if (normalized_dims == 2) {
        // HW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, C, TILE_HEIGHT, Ht * Wt * TILE_WIDTH)
        adjusted_output_grad_shape[2] = TILE_HEIGHT;
        adjusted_output_grad_shape[3] = (output_grad_shape[2] / TILE_HEIGHT) * output_grad_shape[3];
    } else if (normalized_dims == 3) {
        // CHW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (N, 1, TILE_HEIGHT, C * Ht * Wt * TILE_WIDTH)
        adjusted_output_grad_shape[1] = 1;
        adjusted_output_grad_shape[2] = TILE_HEIGHT;
        adjusted_output_grad_shape[3] =
            output_grad_shape[1] * (output_grad_shape[2] / TILE_HEIGHT) * output_grad_shape[3];
    } else if (normalized_dims == 4) {
        // NCHW
        // (N, C, Ht * TILE_HEIGHT, Wt * TILE_WIDTH) -> (1, 1, TILE_HEIGHT, N * C * Ht * Wt * TILE_WIDTH)
        adjusted_output_grad_shape[0] = 1;
        adjusted_output_grad_shape[1] = 1;
        adjusted_output_grad_shape[2] = TILE_HEIGHT;
        adjusted_output_grad_shape[3] =
            output_grad_shape[0] * output_grad_shape[1] * (output_grad_shape[2] / TILE_HEIGHT) * output_grad_shape[3];
    } else {
        TT_ASSERT(is_lastdim_layernorm);
    }

    const auto N = adjusted_output_grad_shape[0];
    const auto C = adjusted_output_grad_shape[1];
    const auto H = adjusted_output_grad_shape[2];
    const auto W = adjusted_output_grad_shape[3];

    const auto Ht = H / TILE_HEIGHT;
    const auto Wt = W / TILE_WIDTH;  // inner_size

    const auto NCHt = N * C * Ht;  // outer_size

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();
    TT_ASSERT(gamma_grad_has_value || beta_grad_has_value);

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord = {core_grid.x_, num_cores_y};

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_cols_per_core_group_1,
         num_cols_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, Wt);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // scaler
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h

    const uint32_t out0_t = 1;  // gamma_grad(==dgamma)
    const uint32_t out1_t = 1;  // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},        // output_grad(==dy)
            {CB::c_in1, in1_t},        // input(==x)
            {CB::c_in2, in2_t},        // mean
            {CB::c_in3, in3_t},        // rstd
            {CB::c_in4, in4_t},        // scaler
            {CB::c_in5, in5_t},        // mask_h
            {CB::c_out0, out0_t},      // gamma_grad(==dgamma)
            {CB::c_out1, out1_t},      // beta_grad(==dbeta)
            {CB::c_intermed0, im0_t},  // output(==y)
            {CB::c_intermed1, im1_t},  // y * dy
            {CB::c_intermed2, im2_t},  // Add[dy]
            {CB::c_intermed3, im3_t},  // Add[y * dy]
            {CB::c_intermed4, im4_t},  // x - mean
            {CB::c_intermed5, im5_t},  // dycopy
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(output_grad)),
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(mean)),
        static_cast<uint32_t>(is_dram(rstd)),
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(do_mask_h)};

    const std::vector<uint32_t> writer_compile_time_args{
        static_cast<uint32_t>(is_dram(gamma_grad)),
        static_cast<uint32_t>(is_dram(beta_grad)),
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value)};

    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "reader_moreh_layernorm_backward_gamma_beta_grad.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "writer_moreh_layernorm_backward_gamma_beta_grad.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> compute_defines{};
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_COL";

    const std::vector<uint32_t> compute_args_group_1{
        num_cols_per_core_group_1,
        origin_H,
        origin_W,
        NCHt,
        Wt,
        static_cast<uint32_t>(gamma_grad_has_value),
        static_cast<uint32_t>(beta_grad_has_value),
        static_cast<uint32_t>(is_lastdim_layernorm),
        static_cast<uint32_t>(is_groupnorm)};

    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_layernorm_backward/kernels/"
        "moreh_layernorm_backward_gamma_beta_grad_kernel.cpp";

    CreateComputeKernel(
        program, compute_kernel_file, {core_group_1, num_cols_per_core_group_1, compute_args_group_1}, compute_defines);

    if (!core_group_2.ranges().empty()) {
        const std::vector<uint32_t> compute_args_group_2{
            num_cols_per_core_group_2,
            origin_H,
            origin_W,
            NCHt,
            Wt,
            static_cast<uint32_t>(gamma_grad_has_value),
            static_cast<uint32_t>(beta_grad_has_value),
            static_cast<uint32_t>(is_lastdim_layernorm),
            static_cast<uint32_t>(is_groupnorm)};

        CreateComputeKernel(
            program,
            compute_kernel_file,
            {core_group_2, num_cols_per_core_group_2, compute_args_group_2},
            compute_defines);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_addr = input.buffer()->address();
    const auto mean_addr = mean.buffer()->address();
    const auto rstd_addr = rstd.buffer()->address();

    const auto gamma_grad_addr = gamma_grad_has_value ? gamma_grad->get().buffer()->address() : 0;
    const auto beta_grad_addr = beta_grad_has_value ? beta_grad->get().buffer()->address() : 0;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        const std::vector<uint32_t> reader_runtime_args{
            output_grad_addr, input_addr, mean_addr, rstd_addr, num_cols_per_core, NCHt, Wt, tile_offset, mask_h};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        const std::vector<uint32_t> writer_runtime_args{
            gamma_grad_addr, beta_grad_addr, num_cols_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        tile_offset += num_cols_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto output_grad_buffer = input_buffers.at(0);
        auto input_buffer = input_buffers.at(1);
        auto mean_buffer = input_buffers.at(2);
        auto rstd_buffer = input_buffers.at(3);

        auto gamma_grad_buffer = input_buffers.at(4);
        auto beta_grad_buffer = input_buffers.at(5);
        TT_ASSERT(gamma_grad_buffer != nullptr || beta_grad_buffer != nullptr);

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = output_grad_buffer->address();
                runtime_args[1] = input_buffer->address();
                runtime_args[2] = mean_buffer->address();
                runtime_args[3] = rstd_buffer->address();
                SetRuntimeArgs(program, reader_kernels_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                if (gamma_grad_buffer != nullptr) {
                    runtime_args[0] = gamma_grad_buffer->address();
                }
                if (beta_grad_buffer != nullptr) {
                    runtime_args[1] = beta_grad_buffer->address();
                }
                SetRuntimeArgs(program, writer_kernels_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
