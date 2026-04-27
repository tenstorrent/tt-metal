// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Ungroups expert outputs back to dense per-token layout, fused with the
// per-token top-K weight scaling.
//
// Inputs (most are direct outputs of moe_group / the same tensors that fed it):
//   expert_out      : [1, 1, T_cap, H]      TILE bf16   — FFN output, packed-by-expert
//   plan            : [1, 1, 1, T_cap]      uint32      — moe_group's plan (flat src row)
//   offsets         : [1, 1, 1, E_local+1]  uint32      — moe_group's offsets
//   counts          : [1, 1, 1, E_local]    uint32      — moe_group's counts
//   metadata        : [D, B, S, K]          uint16      — top-K expert ids per source token
//   scores          : [D, B, S, K]          bf16        — top-K weights per source token
//   local_expert_ids: [E_local]             uint16      — global ids of this device's experts
//
// Output:
//   ungrouped: [D, B, S, H]  ROW_MAJOR bf16 — dense per-token MoE output (per-device).
//
// Algorithm (see tt-train/docs/moe_ungroup_plan.md):
//   - Outer loop over experts e in 0..E_local. Within one expert, the input
//     rows [offsets[e], offsets[e]+counts[e]) target pairwise distinct output
//     tokens (a token's top-K contains e at most once), so cores can scatter
//     in parallel without atomics.
//   - Cross-expert collisions (a token whose top-K hits multiple local
//     experts) are handled by inter-expert barriers: cores RMW-accumulate
//     their per-expert contributions into the output, expert-by-expert.
//   - Expert 0 writes (no DRAM read); subsequent experts RMW. Output is
//     pre-zeroed by the kernel before the per-expert loop.
ttnn::Tensor moe_ungroup(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& counts,
    const ttnn::Tensor& metadata,
    const ttnn::Tensor& scores,
    const ttnn::Tensor& local_expert_ids,
    uint32_t e_local,
    uint32_t k,
    uint32_t d,
    uint32_t b,
    uint32_t s);

}  // namespace ttml::metal
