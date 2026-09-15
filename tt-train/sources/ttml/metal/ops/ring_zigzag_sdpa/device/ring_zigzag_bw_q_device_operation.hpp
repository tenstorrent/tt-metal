// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ring_zigzag_sdpa: the ring_sdpa_fw / ring_sdpa_bw mesh wrapper over the
// single-chip SDPA kernels, with the chips that run chosen by where this
// step's visiting chunk comes from (ops::ZigzagVisitor) instead of by the
// contiguous layout's causal rule. The kernels and the reference ring ops
// are untouched; this exists so the zigzag layout can use them on one chunk
// pair per launch.

#pragma once

#include "metal/ttnn_all_includes.hpp"
#include "ring_zigzag_bw_q_device_operation_types.hpp"
#include "ring_zigzag_bw_q_program_factory.hpp"

namespace ttml::metal::ops::ring_zigzag_bw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

}  // namespace ttml::metal::ops::ring_zigzag_bw

namespace ttml::metal::ops::ring_zigzag_bw::q {

struct RingZigzagBwQDeviceOperation {
    using operation_attributes_t = q::operation_attributes_t;
    using tensor_args_t = q::tensor_args_t;
    using tensor_return_value_t = q::tensor_return_value_t;
    using spec_return_value_t = q::spec_return_value_t;
    using program_factory_t = std::variant<RingZigzagBwQProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_zigzag_bw::q

namespace ttnn::prim {

ttml::metal::ops::ring_zigzag_bw::q::RingZigzagBwQDeviceOperation::tensor_return_value_t ttml_ring_zigzag_bw_q(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& attn_output,
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& intermediates,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    ttml::metal::AttentionMaskType mask_type = ttml::metal::AttentionMaskType::None,
    ttml::metal::ops::ring_zigzag_bw::RingDirection ring_direction =
        ttml::metal::ops::ring_zigzag_bw::RingDirection::Backward,
    ttml::metal::ops::ZigzagVisitor visitor = ttml::metal::ops::ZigzagVisitor::Any,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt);

}  // namespace ttnn::prim
