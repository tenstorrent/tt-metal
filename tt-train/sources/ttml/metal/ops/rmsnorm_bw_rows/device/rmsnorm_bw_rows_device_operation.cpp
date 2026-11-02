// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rmsnorm_bw_rows_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::rmsnorm_bw_rows::device {

void RMSNormBackwardRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const DeviceTensorRequirements dram_bf16{.buffer_type = tt::tt_metal::BufferType::DRAM};
    check_device_tensor(tensor_args.input, "RMSNormBackwardRows", "input", dram_bf16);
    check_device_tensor(tensor_args.gamma, "RMSNormBackwardRows", "gamma", dram_bf16);
    check_device_tensor(tensor_args.rms, "RMSNormBackwardRows", "rms", dram_bf16);
    check_device_tensor(tensor_args.dL_dout, "RMSNormBackwardRows", "dL_dout", dram_bf16);
    if (tensor_args.residual_grad.has_value()) {
        check_device_tensor(*tensor_args.residual_grad, "RMSNormBackwardRows", "residual_grad", dram_bf16);
        TT_FATAL(
            tensor_args.residual_grad->logical_shape() == tensor_args.input.logical_shape(),
            "RMSNormBackwardRows: residual_grad {} must match input {}",
            tensor_args.residual_grad->logical_shape(),
            tensor_args.input.logical_shape());
    }
    const auto& xs = tensor_args.input.logical_shape();
    TT_FATAL(xs.rank() == 4, "RMSNormBackwardRows: input must be 4D, got {}", xs);
    TT_FATAL(
        tensor_args.dL_dout.logical_shape() == xs,
        "RMSNormBackwardRows: dL_dout {} must match input {}",
        tensor_args.dL_dout.logical_shape(),
        xs);
    TT_FATAL(
        tensor_args.gamma.logical_shape()[-1] == xs[-1] && tensor_args.gamma.logical_shape().volume() == xs[-1],
        "RMSNormBackwardRows: gamma {} must be [1, 1, 1, C] for input {}",
        tensor_args.gamma.logical_shape(),
        xs);
    TT_FATAL(
        xs[-1] % tt::constants::TILE_WIDTH == 0,
        "RMSNormBackwardRows: C must be a multiple of 32 (no column mask), got {}",
        xs[-1]);
    TT_FATAL(
        !args.transposed_grad || xs[0] * xs[1] == 1,
        "RMSNormBackwardRows: the transposed gradient needs a [1, 1, S, C] input, got {}",
        xs);
    const uint32_t Wt = xs[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        args.num_slices >= 1 && args.slice_tiles >= 1 && (args.num_slices - 1U) * args.slice_tiles < Wt &&
            args.num_slices * args.slice_tiles >= Wt,
        "RMSNormBackwardRows: split ({} x {}) does not cover Wt = {}",
        args.num_slices,
        args.slice_tiles,
        Wt);
}

spec_return_value_t RMSNormBackwardRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.input;
    const auto& shape = in.logical_shape();
    const tt::tt_metal::TensorLayout layout(
        in.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), in.memory_config());
    spec_return_value_t specs{tt::tt_metal::TensorSpec(shape, layout)};
    if (args.transposed_grad) {
        specs.emplace_back(ttnn::Shape{shape[0], shape[1], shape[3], shape[2]}, layout);
    }
    return specs;
}

tensor_return_value_t RMSNormBackwardRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t out;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        out.push_back(ttnn::create_device_tensor(spec, tensor_args.input.device()));
    }
    return out;
}

ttsl::hash::hash_t RMSNormBackwardRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<RMSNormBackwardRowsDeviceOperation>(
        args.num_slices,
        args.slice_tiles,
        args.transposed_grad,
        tensor_args.residual_grad.has_value(),
        tensor_args.input.dtype(),
        tensor_args.input.logical_shape());
}

}  // namespace ttml::metal::ops::rmsnorm_bw_rows::device

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_rmsnorm_bw_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& gamma,
    const ttnn::Tensor& rms,
    const ttnn::Tensor& dL_dout,
    uint32_t num_slices,
    uint32_t slice_tiles,
    const std::optional<ttnn::Tensor>& residual_grad,
    bool transposed_grad) {
    using OperationType = ttml::metal::ops::rmsnorm_bw_rows::device::RMSNormBackwardRowsDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .num_slices = num_slices, .slice_tiles = slice_tiles, .transposed_grad = transposed_grad};
    auto tensor_args = OperationType::tensor_args_t{
        .input = input, .gamma = gamma, .rms = rms, .dL_dout = dL_dout, .residual_grad = residual_grad};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
