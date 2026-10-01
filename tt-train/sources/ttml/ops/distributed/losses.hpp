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
// dimension across TP devices and communicating only small [B,1,S,1] correction
// tensors (local max, local exp-sum, local target-logit).
//
// Forward (all float32 for numerical stability):
//   1. Local max per device  [B,1,S,1]  → all-gather → global max
//   2. Shifted exp + local sum           → all-reduce  → global sum
//   3. log_normalizer = global_max + log(global_sum)
//   4. On-device one-hot via arange + eq → target logit → all-reduce
//   5. per-position loss = log_normalizer − target_logit  [B,1,S,1]
//   6. ReduceType::MEAN: collapse to scalar via mean over B*S.
//      ReduceType::NONE: return the per-position [B,1,S,1] tensor as-is, so
//      callers can apply per-token weighting (e.g. SFT loss masks) before
//      taking their own reduction.
//
// Backward: purely local, no inter-device communication.
//   dL/dx_k = (softmax_k − onehot_k) * scale * grad_output
//   scale = 1/N for MEAN, 1.0 for NONE.  The trailing multiply broadcasts
//   the upstream gradient: scalar [1,1,1,1] for MEAN, per-position
//   [B,1,S,1] for NONE — both expand against scaled_softmax [B,1,S,V/tp].
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
