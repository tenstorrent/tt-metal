// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "autograd/tensor.hpp"

namespace ttml::ops {

// Chunked gated delta rule (Gated DeltaNet) with a fused backward.
//
//   q, k   [B, 1, T, HK*K]  L2-normalized along K (the recurrence diverges otherwise)
//   v      [B, 1, T, HV*V]
//   g, beta [B, 1, T, HV]   log-space decay gate and write strength
//   initial_state [B, HV, K, V], optional (zeros)
// All of q, k, v, g, beta are BFLOAT16.  HK must divide HV; value head h reads key head h / (HV / HK),
// so the key heads are never repeated.  T must be a multiple of chunk_size (32 or 64).
//
// Returns the output head-major as [B*HV, 1, T, V] BFLOAT16, row b*HV + h.  The forward keeps its
// recurrent states, corrected values and UT inverses for the backward instead of recomputing them;
// the backward also allocates a transient DRAM scratch that grows with B*HV*T (see
// metal::chunk_gated_delta_rule_bw).
autograd::TensorPtr chunk_gated_delta_rule(
    const autograd::TensorPtr& q,
    const autograd::TensorPtr& k,
    const autograd::TensorPtr& v,
    const autograd::TensorPtr& g,
    const autograd::TensorPtr& beta,
    uint32_t key_head_dim,
    uint32_t chunk_size = 64U,
    std::optional<float> scale = std::nullopt,
    const autograd::TensorPtr& initial_state = nullptr);

}  // namespace ttml::ops
