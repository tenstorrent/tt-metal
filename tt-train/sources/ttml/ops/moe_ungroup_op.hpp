// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "autograd/tensor.hpp"
#include "metal/ttnn_all_includes.hpp"

namespace ttml::ops {

// Autograd wrapper around metal::moe_ungroup.
//
// Forward: scatters expert outputs back to dense per-token layout, fused
// with the per-token routing-weight scaling baked into `grouped_scores`:
//
//   ungrouped[t, h] = sum_{i: plan[i] == t} grouped_scores[i] * expert_out[i, h]
//
// Backward routes upstream `d(ungrouped) [D,B,S,H]` to:
//   d(expert_out)     [1,1,T_cap,H] = grouped_scores[i] * grad_grouped[i]
//   d(grouped_scores) [1,1,1,T_cap] = sum_h expert_out[i, h] * grad_grouped[i, h]
//
// where `grad_grouped = metal::moe_group(d(ungrouped), …)` — i.e. we
// reuse moe_group as the pure-gather inverse of moe_ungroup. No new
// kernels.
autograd::TensorPtr moe_ungroup_op(
    const autograd::TensorPtr& expert_out,      // [1, 1, T_cap, H]      TILE bf16
    const autograd::TensorPtr& grouped_scores,  // [1, 1, 1, T_cap]      ROW_MAJOR bf16
    const ttnn::Tensor& metadata,               // [D, B, S, K]          uint16  raw
    const ttnn::Tensor& local_expert_ids,       // [E_local]             uint16  raw
    const ttnn::Tensor& plan,                   // [1, 1, 1, T_cap]      uint32  raw
    const ttnn::Tensor& offsets,                // [1, 1, 1, E_local+1]  uint32  raw
    uint32_t e_local,
    uint32_t k,
    uint32_t d,
    uint32_t b,
    uint32_t s);

}  // namespace ttml::ops
