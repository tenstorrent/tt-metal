// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "add_rmsnorm_fw_rows_device_operation.hpp"

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::add_rmsnorm_fw_rows::device {

void AddRMSNormForwardRowsDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const DeviceTensorRequirements dram_bf16{.buffer_type = tt::tt_metal::BufferType::DRAM};
    check_device_tensor(tensor_args.residual, "AddRMSNormForwardRows", "residual", dram_bf16);
    check_device_tensor(tensor_args.delta, "AddRMSNormForwardRows", "delta", dram_bf16);
    check_device_tensor(tensor_args.gamma, "AddRMSNormForwardRows", "gamma", dram_bf16);
    const auto& xs = tensor_args.residual.logical_shape();
    TT_FATAL(xs.rank() == 4, "AddRMSNormForwardRows: residual must be 4D, got {}", xs);
    if (args.delta_transposed) {
        const ttnn::Shape delta_t_shape({xs[0], xs[1], xs[3], xs[2]});
        TT_FATAL(
            xs[0] * xs[1] == 1 && tensor_args.delta.logical_shape() == delta_t_shape,
            "AddRMSNormForwardRows: transposed delta {} must be the [1, 1, C, S] transpose of residual {}",
            tensor_args.delta.logical_shape(),
            xs);
    } else {
        TT_FATAL(
            tensor_args.delta.logical_shape() == xs,
            "AddRMSNormForwardRows: delta {} must match residual {}",
            tensor_args.delta.logical_shape(),
            xs);
    }
    TT_FATAL(
        tensor_args.gamma.logical_shape()[-1] == xs[-1] && tensor_args.gamma.logical_shape().volume() == xs[-1],
        "AddRMSNormForwardRows: gamma {} must be [1, 1, 1, C] for residual {}",
        tensor_args.gamma.logical_shape(),
        xs);
    TT_FATAL(
        xs[-1] % tt::constants::TILE_WIDTH == 0,
        "AddRMSNormForwardRows: C must be a multiple of 32 (no column mask), got {}",
        xs[-1]);
    const uint32_t Wt = xs[-1] / tt::constants::TILE_WIDTH;
    TT_FATAL(
        args.num_slices >= 1 && args.slice_tiles >= 1 && (args.num_slices - 1U) * args.slice_tiles < Wt &&
            args.num_slices * args.slice_tiles >= Wt,
        "AddRMSNormForwardRows: split ({} x {}) does not cover Wt = {}",
        args.num_slices,
        args.slice_tiles,
        Wt);
}

spec_return_value_t AddRMSNormForwardRowsDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& in = tensor_args.residual;
    const auto& shape = in.logical_shape();
    const tt::tt_metal::TensorLayout layout(
        in.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), in.memory_config());
    const ttnn::Shape out_shape{
        shape[0], shape[1], shape[2], shape[3] + args.out_tail_tiles * tt::constants::TILE_WIDTH};
    return {
        tt::tt_metal::TensorSpec(shape, layout),
        tt::tt_metal::TensorSpec(out_shape, layout),
        tt::tt_metal::TensorSpec(ttnn::Shape{shape[0], shape[1], shape[2], 1}, layout)};
}

tensor_return_value_t AddRMSNormForwardRowsDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    tensor_return_value_t out;
    for (const auto& spec : compute_output_specs(args, tensor_args)) {
        out.push_back(ttnn::create_device_tensor(spec, tensor_args.residual.device()));
    }
    return out;
}

ttsl::hash::hash_t AddRMSNormForwardRowsDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<AddRMSNormForwardRowsDeviceOperation>(
        args.num_slices,
        args.slice_tiles,
        args.epsilon,
        args.delta_transposed,
        args.out_tail_tiles,
        tensor_args.residual.dtype(),
        tensor_args.residual.logical_shape());
}

}  // namespace ttml::metal::ops::add_rmsnorm_fw_rows::device

namespace ttnn::prim {

std::vector<ttnn::Tensor> ttml_add_rmsnorm_fw_rows(
    const ttnn::Tensor& residual,
    const ttnn::Tensor& delta,
    const ttnn::Tensor& gamma,
    uint32_t num_slices,
    uint32_t slice_tiles,
    float epsilon,
    bool delta_transposed,
    uint32_t out_tail_tiles) {
    using OperationType = ttml::metal::ops::add_rmsnorm_fw_rows::device::AddRMSNormForwardRowsDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .num_slices = num_slices,
        .slice_tiles = slice_tiles,
        .epsilon = epsilon,
        .delta_transposed = delta_transposed,
        .out_tail_tiles = out_tail_tiles};
    auto tensor_args = OperationType::tensor_args_t{.residual = residual, .delta = delta, .gamma = gamma};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
