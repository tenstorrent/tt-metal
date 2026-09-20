// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_softmax_merge_device_operation.hpp"

#include <tt-metalium/host_api.hpp>

#include "metal/ops/common/ring_sdpa_utils.hpp"

namespace ttml::metal::ops::ring_softmax_merge {

void RingSoftmaxMergeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    TT_FATAL(attrs.ring_size > 0 && attrs.step < attrs.ring_size, "ring step {} of {}", attrs.step, attrs.ring_size);
    for (const auto* tensor : {&t.out_acc, &t.lse_acc, &t.step_out, &t.step_lse}) {
        TT_FATAL(tensor->storage_type() == ttnn::StorageType::DEVICE, "ring_softmax_merge takes device tensors");
        TT_FATAL(tensor->layout() == ttnn::Layout::TILE, "ring_softmax_merge takes tile-layout tensors");
    }
    const auto shape = t.out_acc.logical_shape();
    TT_FATAL(shape.rank() == 4U, "ring_softmax_merge takes (B, H, S, d) tensors");
    TT_FATAL(
        t.out_acc.dtype() == ttnn::DataType::FLOAT32 && t.lse_acc.dtype() == ttnn::DataType::FLOAT32 &&
            t.step_lse.dtype() == ttnn::DataType::FLOAT32,
        "ring_softmax_merge keeps the accumulators and the lse in Float32");
    TT_FATAL(
        t.step_out.dtype() == ttnn::DataType::BFLOAT16,
        "ring_softmax_merge takes the step's output in bfloat16 (as the forward kernels produce it)");
    TT_FATAL(
        t.step_out.logical_shape() == shape,
        "the step's output {} must have the accumulator's shape {}",
        t.step_out.logical_shape(),
        shape);
    const ttnn::Shape lse_shape{shape[0], shape[1], shape[2], 32U};
    TT_FATAL(
        t.lse_acc.logical_shape() == lse_shape && t.step_lse.logical_shape() == lse_shape,
        "the lse tensors must be (B, H, S, 32) with the value in column 0; got {} and {}, expected {}",
        t.lse_acc.logical_shape(),
        t.step_lse.logical_shape(),
        lse_shape);
    TT_FATAL(shape[3] % 32U == 0U && shape[2] % 32U == 0U, "whole tiles only");
}

RingSoftmaxMergeDeviceOperation::spec_return_value_t RingSoftmaxMergeDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& t) {
    return {t.out_acc.tensor_spec(), t.lse_acc.tensor_spec()};
}

RingSoftmaxMergeDeviceOperation::tensor_return_value_t RingSoftmaxMergeDeviceOperation::create_output_tensors(
    const operation_attributes_t&, const tensor_args_t& t) {
    // In place: the accumulators are the outputs.
    return {t.out_acc, t.lse_acc};
}

ttsl::hash::hash_t RingSoftmaxMergeDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    return ttsl::hash::hash_objects(
        attrs.ring_size,
        attrs.ring_axis,
        attrs.step,
        static_cast<int>(attrs.ring_direction),
        attrs.zigzag,
        static_cast<int>(attrs.visitor),
        attrs.mask_type,
        t.out_acc.logical_shape(),
        t.step_out.dtype());
}

}  // namespace ttml::metal::ops::ring_softmax_merge

namespace ttnn::prim {

ttml::metal::ops::ring_softmax_merge::RingSoftmaxMergeDeviceOperation::tensor_return_value_t ttml_ring_softmax_merge(
    const ttnn::Tensor& out_acc,
    const ttnn::Tensor& lse_acc,
    const ttnn::Tensor& step_out,
    const ttnn::Tensor& step_lse,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::ops::ring_softmax_merge::RingDirection ring_direction,
    bool zigzag,
    ttml::metal::ops::ZigzagVisitor visitor,
    ttml::metal::AttentionMaskType mask_type) {
    using OperationType = ttml::metal::ops::ring_softmax_merge::RingSoftmaxMergeDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .ring_size = ring_size,
        .ring_axis = ring_axis,
        .step = step,
        .ring_direction = ring_direction,
        .zigzag = zigzag,
        .visitor = visitor,
        .mask_type = mask_type};
    auto tensors = OperationType::tensor_args_t{
        .out_acc = out_acc, .lse_acc = lse_acc, .step_out = step_out, .step_lse = step_lse};
    return ttnn::device_operation::launch<OperationType>(attrs, tensors);
}

}  // namespace ttnn::prim
