// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {
using namespace constants;
namespace operations {

namespace primary {

operation::ProgramWithCallbacks moreh_bias_backward_single_core_hw(const Tensor &output_grad, const Tensor &bias_grad) {
    Program program{};
    CoreCoord core = {0, 0};
    const uint32_t core_num = 1;

    DataFormat cb_data_format = datatype_to_dataformat_converter(output_grad.get_dtype());
    uint32_t single_tile_size = detail::TileSize(cb_data_format);

    Buffer *src_buffer = output_grad.buffer();
    const auto &bias_grad_shape = bias_grad.get_legacy_shape().without_padding();
    Buffer *dst_buffer = bias_grad.buffer();
    uint32_t num_tiles = output_grad.volume() / TILE_HW;
    const auto &output_grad_shape_wo_padding = output_grad.get_legacy_shape().without_padding();
    const bool do_mask_h = (output_grad_shape_wo_padding[2] % TILE_HEIGHT) != 0;
    const uint32_t mask_h = do_mask_h ? output_grad_shape_wo_padding[2] % TILE_HEIGHT : TILE_HEIGHT;
    const bool do_mask_w = (output_grad_shape_wo_padding[3] % TILE_WIDTH) != 0;
    const uint32_t mask_w = do_mask_w ? output_grad_shape_wo_padding[3] % TILE_WIDTH : TILE_WIDTH;

    const auto &output_grad_shape = output_grad.get_legacy_shape();
    uint32_t B1 = output_grad_shape[0];
    uint32_t B2 = output_grad_shape[1];
    uint32_t Ht = output_grad_shape[2] / TILE_HEIGHT;
    uint32_t Wt = output_grad_shape[3] / TILE_WIDTH;

    const uint32_t in0_t = 2;
    const uint32_t in1_t = 1;
    const uint32_t in2_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    const uint32_t out0_t = 1;
    const uint32_t im0_t = 1;
    const uint32_t im1_t = 1;

    // This should allocate a DRAM buffer on the device
    Device *device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    CreateCircularBuffer(
        program,
        std::set<CoreRange>{CoreRange(core, core)},
        cb_data_format,
        {
            {CB::c_in0, in0_t},    // output_grad
            {CB::c_in1, in1_t},    // scaler
            {CB::c_in2, in2_t},    // mask_h_w
            {CB::c_out0, out0_t},  // bias_grad
            {CB::c_intermed0, im0_t},
            {CB::c_intermed1, im1_t},
        });

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::vector<uint32_t> reader_compile_time_args{static_cast<uint32_t>(is_dram(output_grad))};
    const std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(bias_grad))};

    const auto reader_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_linear_backward/kernels/reader_moreh_bias_backward_hw.cpp";

    const auto writer_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_linear_backward/kernels/writer_moreh_bias_backward.cpp";

    const auto reader_kernel_id = CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);
    const auto writer_kernel_id = CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    vector<uint32_t> compute_kernel_args = {};
    std::map<string, string> compute_defines;
    compute_defines["REDUCE_OP"] = "PoolType::SUM";
    compute_defines["REDUCE_DIM"] = "ReduceDim::REDUCE_SCALAR";
    const auto compute_kernel_file =
        "tt_eager/tt_dnn/op_library/moreh_linear_backward/kernels/moreh_bias_backward_single_core_hw.cpp";
    const auto compute_kernel_id =
        CreateComputeKernel(program, compute_kernel_file, {core, core_num, compute_kernel_args}, compute_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    SetRuntimeArgs(
        program, reader_kernel_id, core, {src_buffer->address(), num_tiles, 0, mask_h, mask_w, do_mask_h, do_mask_w});
    SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), 1, 0});
    SetRuntimeArgs(program, compute_kernel_id, core, {B1, B2, Ht, Wt, do_mask_h, do_mask_w});

    auto override_runtime_arguments_callback = [reader_kernel_id, writer_kernel_id](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {
        const auto &output_grad = input_tensors.at(0);
        const auto &bias_grad = input_tensors.at(1);

        Buffer *src_buffer = output_grad.buffer();
        Buffer *dst_buffer = bias_grad.buffer();
        CoreCoord core = {0, 0};
        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };
    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
