// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/ring_sdpa_bw_kv_device_operation.hpp"
#include "device/ring_sdpa_bw_q_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

using RingDirection = ttml::metal::ops::ring_sdpa_bw::RingDirection;

// Shifted-query tensors for RingSchedule::Striped (see metal/ops/common/ring_sdpa_utils.hpp).
// Devices whose K/V source comes later in the ring (src > d) read the shifted inputs and write
// grad_query in shifted row coordinates; K/V and the dK/dV outputs are shared with the primary set.
struct RingSDPABwShifted {
    ttnn::Tensor grad_output;
    ttnn::Tensor attn_output;
    ttnn::Tensor query;
    ttnn::Tensor intermediates;
    ttnn::Tensor grad_query;
};

// Returns [grad_Q, grad_K, grad_V] (grad_Q of the primary query)
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_sdpa_bw(
    const ttnn::Tensor& grad_output,    // Gradient w.r.t. output
    const ttnn::Tensor& attn_output,    // sdpa forward output (needed for gradients)
    const ttnn::Tensor& query,          // input Q (needed for gradients)
    const ttnn::Tensor& key,            // input K (needed for gradients)
    const ttnn::Tensor& value,          // input V (needed for gradients)
    const ttnn::Tensor& intermediates,  // From forward pass (FP32 logsumexp per row)
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type = AttentionMaskType::None,
    RingDirection ring_direction = RingDirection::Backward,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt,
    ops::RingSchedule schedule = ops::RingSchedule::Contiguous,
    const std::optional<RingSDPABwShifted>& shifted = std::nullopt);

}  // namespace ttml::metal
