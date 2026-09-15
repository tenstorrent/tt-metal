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
#include "ring_zigzag_bw_kv_device_operation_types.hpp"
#include "ring_zigzag_bw_kv_program_factory.hpp"

namespace ttml::metal::ops::ring_zigzag_bw {

using RingDirection = ttnn_fixed::distributed::RingShiftDirection;

}  // namespace ttml::metal::ops::ring_zigzag_bw

namespace ttml::metal::ops::ring_zigzag_bw::kv {

struct RingZigzagBwKVDeviceOperation {
    using operation_attributes_t = kv::operation_attributes_t;
    using tensor_args_t = kv::tensor_args_t;
    using tensor_return_value_t = kv::tensor_return_value_t;
    using spec_return_value_t = kv::spec_return_value_t;
    using program_factory_t = std::variant<RingZigzagBwKVProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttml::metal::ops::ring_zigzag_bw::kv

namespace ttnn::prim {

ttml::metal::ops::ring_zigzag_bw::kv::RingZigzagBwKVDeviceOperation::tensor_return_value_t ttml_ring_zigzag_bw_kv(
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& u_scaler,
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
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt);

}  // namespace ttnn::prim
