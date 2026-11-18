// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "autograd/tensor.hpp"
#include "ops/losses.hpp"

namespace ttml::ops::distributed {

// Vocab-parallel cross-entropy loss.
//
// Avoids the full-vocabulary gather by keeping logits sharded along the vocab
// dimension across TP devices and communicating only one small FP32 record
// per row and shard.
//
// Collapse-resistant (arXiv:2610.31642, Section 2), like the fused single-device
// cross-entropy: the loss of a confidently predicted token is log1p(tail) and its
// gradient -s * tail / (1 + tail), never rounded to zero while tail is representable.
//
// Forward:
//   1. Per shard and row (fused kernel): local maximum, global column of its
//      first occurrence, tail sum_{j != k} exp(x_j - m) without it, and the
//      target logit if this shard owns the target, all FP32.
//   2. All-gather the records across TP (exact; no reduction in the CCL).
//   3. Every device combines them identically: global maximum M, one global first
//      maximum K (lowest shard on ties), T = tail without exactly x_K,
//      l = log1p(T) in FP32, per-position loss = -((x_t - M) - l).
//   4. ReduceType::MEAN: collapse to scalar via mean over B*S.
//      ReduceType::NONE: return the per-position [B,1,S,1] tensor as-is, so
//      callers can apply per-token weighting (e.g. SFT loss masks) before
//      taking their own reduction.
//
// Backward: purely local, no inter-device communication, from the row state (M, l).
//   dL/dx_j = s * exp(x_j - M - l), and on the target's shard
//   dL/dx_t = s * expm1((x_t - M) - l) = s * (softmax_t - 1).
//   s = scale * grad_output: scale = 1/N for MEAN, 1.0 for NONE; the upstream
//   gradient (scalar [1,1,1,1] for MEAN, per-position [B,1,S,1] for NONE) is
//   folded into the kernel's per-row scale.
//
// logits  : [B, 1, S, local_V] bfloat16, sharded along vocab (last) dim across TP.
// targets : [B, 1, S, 1] or [B, S] uint32, integer target indices in [0, full_V),
//           replicated across TP devices within each DP group.
// cluster_axis : mesh axis that logits are sharded across (TP axis).
//                nullopt for a 1-D mesh where all devices are TP.
// reduce  : ReduceType::MEAN (default) or ReduceType::NONE.  ReduceType::SUM
//           is rejected to match the ttml::ops::cross_entropy_loss contract.
autograd::TensorPtr vocab_parallel_cross_entropy_loss(
    const autograd::TensorPtr& logits,
    const autograd::TensorPtr& targets,
    std::optional<uint32_t> cluster_axis = std::nullopt,
    ReduceType reduce = ReduceType::MEAN);

}  // namespace ttml::ops::distributed
