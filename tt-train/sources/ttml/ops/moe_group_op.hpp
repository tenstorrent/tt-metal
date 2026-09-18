// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::ops {

// Outputs of moe_group_op. `grouped` and `grouped_scores` are autograd
// tensors — gradients flow back to `dispatched` and `scores` respectively
// via metal::moe_ungroup in the backward. The remaining four (k_slot,
// counts, offsets, plan) are pure routing metadata; no gradient flows
// through them.
struct MoEGroupOutputs {
    autograd::TensorPtr grouped;         // [1, 1, T_cap, H]      TILE bf16
    autograd::TensorPtr grouped_scores;  // [1, 1, 1, T_cap]      ROW_MAJOR bf16
    ttnn::Tensor k_slot;                 // [1, 1, 1, T_cap]      uint16
    ttnn::Tensor counts;                 // [1, 1, 1, E_local]    uint32
    ttnn::Tensor offsets;                // [1, 1, 1, E_local+1]  uint32
    ttnn::Tensor plan;                   // [1, 1, 1, T_cap]      uint32
};

// Autograd wrapper around metal::moe_group.
//
// Forward: gathers tokens from `dispatched [D,B,S,H]` into expert-grouped
// rows `grouped [1,1,T_cap,H]`, and gathers the per-token routing weights
// from `scores [D,B,S,K]` into `grouped_scores [1,1,1,T_cap]` (one bf16
// per active row).
//
// Backward: produces both `d(dispatched)` and `d(scores)` via two calls
// to metal::moe_ungroup with `grouped_scores = ones`, so no per-token
// scaling is applied:
//   - d(dispatched) [D,B,S,H]  ← row-scatter of d(grouped) with H = hidden_dim
//   - d(scores)     [D,B,S,K]  ← K-wide sparse-scatter of d(grouped_scores)
//                                with H = K, where row i contributes
//                                d(grouped_scores)[i] at column k_slot[i]
//                                and zero elsewhere.
MoEGroupOutputs moe_group_op(
    const autograd::TensorPtr& dispatched,  // [D, B, S, H]  bf16  ROW_MAJOR
    const ttnn::Tensor& metadata,           // [D, B, S, K]  uint16  raw
    const autograd::TensorPtr& scores,      // [D, B, S, K]  bf16  ROW_MAJOR
    const ttnn::Tensor& local_expert_ids,   // [E_local]     uint16  raw
    uint32_t e_local,
    uint32_t k);

}  // namespace ttml::ops
