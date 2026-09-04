// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Computes per-position cross-entropy loss: out[n, h] = -input[n, 0, h, target[n, h]] + logsumexp(input[n, 0, h, :]).
//
// Targets must lie in [0, W). A target outside that range is not gathered (no out-of-bounds
// read); its position contributes a zero logit, i.e. loss = logsumexp for that row — the same
// convention select_target_logit uses for out-of-shard targets. There is no ignore_index:
// every position in the logical shape contributes to the loss.
ttnn::Tensor cross_entropy_fw(
    const ttnn::Tensor& input,  // logits : model output (N, 1, H, W)
    const ttnn::Tensor& target  // target : ground truth (N, H)
);

}  // namespace ttml::metal
