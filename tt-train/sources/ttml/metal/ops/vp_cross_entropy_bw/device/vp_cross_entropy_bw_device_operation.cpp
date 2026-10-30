// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "vp_cross_entropy_bw_device_operation.hpp"

#include <limits>

#include "metal/common/tensor_validation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttml::metal::ops::vp_cross_entropy_bw::device {

void VpCrossEntropyBwDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& logits = tensor_args.logits;
    check_device_tensor(logits, "VpCrossEntropyBw", "logits");
    check_device_tensor(
        tensor_args.log_normalizer,
        "VpCrossEntropyBw",
        "log_normalizer",
        {.dtypes = {tt::tt_metal::DataType::FLOAT32}});
    check_device_tensor(
        tensor_args.target,
        "VpCrossEntropyBw",
        "target",
        {.dtypes = {tt::tt_metal::DataType::UINT32}, .layout = tt::tt_metal::Layout::ROW_MAJOR});
    check_device_tensor(
        tensor_args.grad,
        "VpCrossEntropyBw",
        "grad",
        {.dtypes = {tt::tt_metal::DataType::BFLOAT16, tt::tt_metal::DataType::FLOAT32}});
    const auto& ls = logits.logical_shape();
    TT_FATAL(ls.rank() == 4U && ls[1] == 1U, "VpCrossEntropyBw: logits must be [B, 1, S, V], got {}", ls);
    const auto& ns = tensor_args.log_normalizer.logical_shape();
    TT_FATAL(
        ns.rank() == 4U && ns[0] == ls[0] && ns[1] == 1U && ns[2] == ls[2] && ns[3] == 1U,
        "VpCrossEntropyBw: log_normalizer must be [B, 1, S, 1] for logits {}, got {}",
        ls,
        ns);
    const auto& ts = tensor_args.target.logical_shape();
    TT_FATAL(
        ts.rank() == 2U && ts[0] == ls[0] && ts[1] == ls[2],
        "VpCrossEntropyBw: target must be [B, S] for logits {}, got {}",
        ls,
        ts);
    TT_FATAL(
        tensor_args.grad.padded_shape().volume() == tt::constants::TILE_HW,
        "VpCrossEntropyBw: grad must be one tile holding g in every element, got {}",
        tensor_args.grad.logical_shape());
    TT_FATAL(args.local_V > 0U, "VpCrossEntropyBw: local_V must be > 0");
    if (args.cluster_axis.has_value()) {
        TT_FATAL(
            *args.cluster_axis < logits.device()->shape().dims(),
            "VpCrossEntropyBw: cluster_axis {} out of range",
            *args.cluster_axis);
    }
}

spec_return_value_t VpCrossEntropyBwDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    return tensor_args.logits.tensor_spec();
}

tensor_return_value_t VpCrossEntropyBwDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return ttnn::create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.logits.device());
}

ttsl::hash::hash_t VpCrossEntropyBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return tt::tt_metal::operation::hash_operation<VpCrossEntropyBwDeviceOperation>(
        args.cluster_axis.value_or(std::numeric_limits<uint32_t>::max()),
        args.first_v,
        args.local_V,
        args.scale,
        tensor_args.logits.logical_shape(),
        tensor_args.grad.dtype());
}

}  // namespace ttml::metal::ops::vp_cross_entropy_bw::device

namespace ttnn::prim {

ttnn::Tensor ttml_vp_cross_entropy_bw(
    const ttnn::Tensor& logits,
    const ttnn::Tensor& log_normalizer,
    const ttnn::Tensor& target,
    const ttnn::Tensor& grad,
    uint32_t local_V,
    std::optional<uint32_t> cluster_axis,
    uint32_t first_v,
    float scale) {
    using OperationType = ttml::metal::ops::vp_cross_entropy_bw::device::VpCrossEntropyBwDeviceOperation;
    auto attributes = OperationType::operation_attributes_t{
        .first_v = first_v, .local_V = local_V, .cluster_axis = cluster_axis, .scale = scale};
    auto tensor_args = OperationType::tensor_args_t{
        .logits = logits, .log_normalizer = log_normalizer, .target = target, .grad = grad};
    return ttnn::device_operation::launch<OperationType>(attributes, tensor_args);
}

}  // namespace ttnn::prim
