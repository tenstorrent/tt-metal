// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "tt_eager/tensor/tensor_impl.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_eager/tt_numpy/functions.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace operations {

namespace primary {

namespace {
std::tuple<uint32_t, float, bool> get_floored_p_and_decimal_and_p_is_negative(float p) {
    auto floored_p = std::floor(p);
    auto decimal = p - floored_p;
    const bool p_is_negative = floored_p < 0.0f;
    if (p_is_negative) {
        floored_p = -floored_p;
    }
    return std::make_tuple(static_cast<uint32_t>(floored_p), decimal, p_is_negative);
}
}  // namespace

operation::ProgramWithCallbacks moreh_norm_backward_(
    const Tensor &input, const Tensor &output, const Tensor &output_grad, float p, const Tensor &input_grad) {
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto device = output_grad.device();
    auto program = CreateProgram();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // input
    const auto input_shape = input.get_legacy_shape();
    const auto input_rank = static_cast<int64_t>(input_shape.rank());

    const auto input_n = input_shape[0];
    const auto input_c = input_shape[1];
    const auto input_h = input_shape[2];
    const auto input_w = input_shape[3];
    const auto input_ht = input_h / TILE_HEIGHT;
    const auto input_wt = input_w / TILE_WIDTH;

    const auto input_origin_shape = input_shape.without_padding();

    const auto input_origin_h = input_origin_shape[2];
    const auto input_origin_w = input_origin_shape[3];

    // output
    const auto output_shape = output.get_legacy_shape();
    const auto output_rank = static_cast<int64_t>(output_shape.rank());

    const auto output_n = output_shape[0];
    const auto output_c = output_shape[1];
    const auto output_h = output_shape[2];
    const auto output_w = output_shape[3];
    const auto output_ht = output_h / TILE_HEIGHT;
    const auto output_wt = output_w / TILE_WIDTH;

    const auto output_origin_shape = output_shape.without_padding();

    const auto output_origin_h = output_origin_shape[2];
    const auto output_origin_w = output_origin_shape[3];

    const auto need_to_bcast_n = static_cast<uint32_t>(output_n != input_n);
    const auto need_to_bcast_c = static_cast<uint32_t>(output_c != input_c);
    const auto need_to_bcast_ht = static_cast<uint32_t>(output_origin_h != input_origin_h);
    const auto need_to_bcast_wt = static_cast<uint32_t>(output_origin_w != input_origin_w);

    const auto num_input_tiles = input.volume() / TILE_HW;

    auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
    auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
        get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

    TT_ASSERT(tt::numpy::detail::nearly_equal(decimal_minus_one, decimal));

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CoreGridDesc core_grid(device);
    const auto num_cores_y = core_grid.y_;
    CoreCoord core_grid_coord(core_grid.x_, num_cores_y);

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_input_tiles_per_core_group_1,
         num_input_tiles_per_core_group_2] = tt_metal::split_work_to_cores(core_grid_coord, num_input_tiles);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto cb_data_format = tt_metal::datatype_to_dataformat_converter(output_grad.get_dtype());

    const uint32_t in0_t{1};  // input(==x)
    const uint32_t in1_t{1};  // output(==y)
    const uint32_t in2_t{1};  // output_grad(==dy)
    const uint32_t in3_t{1};  // decimal

    // (x^(p - 1) * y * dy) / y^p
    const uint32_t out0_t{1};  // input_grad(==dx)

    const uint32_t im0_t{1};
    const uint32_t im1_t{1};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};

    CreateCircularBuffer(
        program,
        all_cores,
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // input
            {CB::c_in1, in1_t},    // output
            {CB::c_in2, in2_t},    // output_grad
            {CB::c_in3, in3_t},    // decimal
            {CB::c_out0, out0_t},  // input_grad
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
            {CB::c_intermed2, im2_t},
            {CB::c_intermed3, im3_t},
            {CB::c_intermed4, im4_t},
            {CB::c_intermed5, im5_t},
            {CB::c_intermed6, im6_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "reader_moreh_norm_backward.cpp";
    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "writer_moreh_norm_backward.cpp";

    const auto reader_kernels_id = CreateReadKernel(program, reader_kernel_file, all_cores);
    const auto writer_kernels_id = CreateWriteKernel(program, writer_kernel_file, all_cores);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_norm_backward/kernels/"
        "moreh_norm_backward_kernel.cpp";

    const auto compute_kernels_id_1 =
        CreateComputeKernel(program, compute_kernel_file, {core_group_1, num_input_tiles_per_core_group_1});

    KernelHandle compute_kernels_id_2{0};
    if (!core_group_2.ranges().empty()) {
        compute_kernels_id_2 =
            CreateComputeKernel(program, compute_kernel_file, {core_group_2, num_input_tiles_per_core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto input_addr = input.buffer()->address();
    const auto output_addr = output.buffer()->address();
    const auto output_grad_addr = output_grad.buffer()->address();
    const auto input_grad_addr = input_grad.buffer()->address();

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_input_tiles_per_core;
        KernelHandle compute_kernel_id;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_input_tiles_per_core = num_input_tiles_per_core_group_1;
            compute_kernel_id = compute_kernels_id_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_input_tiles_per_core = num_input_tiles_per_core_group_2;
            compute_kernel_id = compute_kernels_id_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        const std::vector<uint32_t> reader_runtime_args{
            input_addr,
            static_cast<uint32_t>(is_dram(input)),
            output_addr,
            static_cast<uint32_t>(is_dram(output)),
            output_grad_addr,
            static_cast<uint32_t>(is_dram(output_grad)),
            *reinterpret_cast<uint32_t *>(&decimal),
            num_input_tiles_per_core,
            tile_offset,
            input_n,
            input_c,
            input_origin_h,
            input_origin_w,
            output_n,
            output_c,
            output_origin_h,
            output_origin_w};
        SetRuntimeArgs(program, reader_kernels_id, core, reader_runtime_args);

        // writer
        const std::vector<uint32_t> writer_runtime_args{
            input_grad_addr, static_cast<uint32_t>(is_dram(input_grad)), num_input_tiles_per_core, tile_offset};
        SetRuntimeArgs(program, writer_kernels_id, core, writer_runtime_args);

        // compute
        const std::vector<uint32_t> compute_runtime_args{
            num_input_tiles_per_core,
            need_to_bcast_n,
            need_to_bcast_c,
            need_to_bcast_ht,
            need_to_bcast_wt,
            floored_p,
            static_cast<uint32_t>(p_is_negative),
            floored_p_minus_one,
            static_cast<uint32_t>(p_minus_one_is_negative)};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);

        tile_offset += num_input_tiles_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Callback SetUp
    ////////////////////////////////////////////////////////////////////////////
    auto override_runtime_args_callback = [reader_kernels_id = reader_kernels_id,
                                           writer_kernels_id = writer_kernels_id,
                                           compute_kernels_id_1 = compute_kernels_id_1,
                                           compute_kernels_id_2 = compute_kernels_id_2,
                                           num_cores_to_be_used = num_cores_to_be_used,
                                           num_cores_y = num_cores_y,
                                           core_group_1 = core_group_1,
                                           core_group_2 = core_group_2](
                                              const void *operation,
                                              Program &program,
                                              const std::vector<Tensor> &input_tensors,
                                              const std::vector<std::optional<const Tensor>> &,
                                              const std::vector<Tensor> &) {
        const auto p = static_cast<const MorehNormBackward *>(operation)->p;

        auto [floored_p, decimal, p_is_negative] = get_floored_p_and_decimal_and_p_is_negative(p);
        auto [floored_p_minus_one, decimal_minus_one, p_minus_one_is_negative] =
            get_floored_p_and_decimal_and_p_is_negative(p - 1.0f);

        TT_ASSERT(tt::numpy::detail::nearly_equal(decimal_minus_one, decimal));

        auto input_buffer = input_tensors.at(0).buffer();
        auto output_buffer = input_tensors.at(1).buffer();
        auto output_grad_buffer = input_tensors.at(2).buffer();
        auto input_grad_buffer = input_tensors.at(3).buffer();

        for (uint32_t i = 0; i < num_cores_to_be_used; ++i) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, reader_kernels_id, core);
                runtime_args[0] = input_buffer->address();
                runtime_args[2] = output_buffer->address();
                runtime_args[4] = output_grad_buffer->address();
                runtime_args[6] = *reinterpret_cast<uint32_t *>(&decimal);
                SetRuntimeArgs(program, reader_kernels_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, writer_kernels_id, core);
                runtime_args[0] = input_grad_buffer->address();
                SetRuntimeArgs(program, writer_kernels_id, core, runtime_args);
            }

            {
                KernelHandle compute_kernel_id;
                if (core_group_1.core_coord_in_core_ranges(core)) {
                    compute_kernel_id = compute_kernels_id_1;
                } else if (core_group_2.core_coord_in_core_ranges(core)) {
                    compute_kernel_id = compute_kernels_id_2;
                } else {
                    TT_THROW("Core not in specified core ranges.");
                }
                auto runtime_args = GetRuntimeArgs(program, compute_kernel_id, core);
                runtime_args[5] = floored_p;
                runtime_args[6] = static_cast<uint32_t>(p_is_negative);
                runtime_args[7] = floored_p_minus_one;
                runtime_args[8] = static_cast<uint32_t>(p_minus_one_is_negative);
                SetRuntimeArgs(program, compute_kernel_id, core, runtime_args);
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace primary

}  // namespace operations

}  // namespace tt
