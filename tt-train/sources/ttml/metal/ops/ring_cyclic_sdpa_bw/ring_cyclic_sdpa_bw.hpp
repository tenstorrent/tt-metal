// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "device/ring_cyclic_sdpa_bw_device_operation.hpp"
#include "metal/common/const_utils.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

using RingCyclicDirection = ttml::metal::ops::ring_cyclic_sdpa_bw::RingDirection;

// One step of a context-parallel ring backward, computed with the cyclic
// schedule instead of the two-pass sdpa_bw.
//
// Drop-in for ring_sdpa_bw in the step loop: same skip pattern, same per-chip
// mask decision, same tensors. What differs is what an executing chip runs --
// one fused kernel over a cyclic schedule rather than a dQ pass and a dK/dV
// pass, each recomputing the score stage.
//
// The gradients are accumulated in place, which is the other difference. The
// caller passes its running FP32 accumulators as the preallocated outputs and
// the kernels add into them, where ring_sdpa_bw returns per-step bf16
// gradients the caller must upcast and add on the host.
//
// Statistics are the *global* log-sum-exp and D = rowsum(dO . O), as the ring
// requires: each step's partial contribution is only correct against the
// global normaliser.
std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor> ring_cyclic_sdpa_bw(
    const ttnn::Tensor& query,
    const ttnn::Tensor& key,
    const ttnn::Tensor& value,
    const ttnn::Tensor& grad_output,
    const ttnn::Tensor& log_sum_exp,
    const ttnn::Tensor& row_scalar,
    uint32_t ring_size,
    uint32_t ring_axis,
    uint32_t step,
    AttentionMaskType mask_type = AttentionMaskType::Causal,
    RingCyclicDirection ring_direction = RingCyclicDirection::Backward,
    uint32_t rows_per_block_tiles = 1U,
    bool use_barrier = false,
    bool accumulate_into_outputs = false,
    const std::optional<ttnn::Tensor>& preallocated_grad_query = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_key = std::nullopt,
    const std::optional<ttnn::Tensor>& preallocated_grad_value = std::nullopt);

}  // namespace ttml::metal
