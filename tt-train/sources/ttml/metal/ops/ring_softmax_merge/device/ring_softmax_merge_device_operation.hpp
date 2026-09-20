// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_softmax_merge_device_operation_types.hpp"
#include "ring_softmax_merge_program_factory.hpp"

namespace ttml::metal::ops::ring_softmax_merge {

struct RingSoftmaxMergeDeviceOperation {
    using operation_attributes_t = ring_softmax_merge::operation_attributes_t;
    using tensor_args_t = ring_softmax_merge::tensor_args_t;
    using spec_return_value_t = ring_softmax_merge::spec_return_value_t;
    using tensor_return_value_t = ring_softmax_merge::tensor_return_value_t;
    using program_factory_t = std::variant<RingSoftmaxMergeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

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
    ttml::metal::AttentionMaskType mask_type);

}  // namespace ttnn::prim
