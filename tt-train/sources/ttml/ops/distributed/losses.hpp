// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops::distributed {

// Sharded cross-entropy loss for vocab-sharded logits.
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
//   5. loss = mean(log_normalizer − target_logit)
//
// Backward: purely local, no inter-device communication.
//   dL/dx_k = (softmax_k − onehot_k) / N * grad_output
//
// logits  : [B, 1, S, local_V] bfloat16, sharded along vocab (last) dim across TP.
// targets : [B, 1, S, 1] or [B, S] uint32, integer target indices in [0, full_V),
//           replicated across TP devices within each DP group.
// cluster_axis : mesh axis that logits are sharded across (TP axis).
//                nullopt for a 1-D mesh where all devices are TP.
autograd::TensorPtr sharded_cross_entropy_loss(
    const autograd::TensorPtr& logits,
    const autograd::TensorPtr& targets,
    std::optional<uint32_t> cluster_axis = std::nullopt);

}  // namespace ttml::ops::distributed
