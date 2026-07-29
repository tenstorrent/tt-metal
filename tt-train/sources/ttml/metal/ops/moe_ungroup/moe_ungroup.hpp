// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Ungroups expert outputs back to dense per-token layout, fused with the
// per-token top-K weight scaling.
//
// Inputs (all direct outputs of moe_group, no metadata scan needed):
//   expert_out      : [1, 1, T_cap, H]      TILE bf16   — FFN output, packed-by-expert
//   plan            : [1, 1, 1, T_cap]      uint32      — moe_group's plan (flat src row)
//   offsets         : [1, 1, 1, E_local+1]  uint32      — moe_group's offsets
//   grouped_scores  : [1, 1, 1, T_cap]      bf16        — moe_group's grouped_scores
//                                                          (= scores[plan[i], k_slot[i]])
//
// ABI invariants:
//   - T_cap is a multiple of 32.
//   - offsets[0] == 0 and every offsets[i] is a multiple of 32. The dataflow
//     kernels divide offsets by TILE_HEIGHT and process whole tile rows; direct
//     callers must preserve the alignment that moe_group produces.
//
// Output:
//   ungrouped: [D, B, S, H]  ROW_MAJOR bf16 — dense per-token MoE output (per-device).
//
// Algorithm:
//   - Outer loop over experts e in 0..E_local. Within one expert, the input
//     rows offsets[e]..offsets[e+1] target pairwise distinct output tokens
//     (a token's top-K contains e at most once), so cores can scatter in
//     parallel without atomics.
//   - Cross-expert collisions (a token whose top-K hits multiple local
//     experts) are handled by inter-expert barriers: cores RMW-accumulate
//     their per-expert contributions into the output, expert-by-expert.
//   - Output is pre-zeroed by the kernel before the per-expert loop, so the
//     first expert to touch a row reads zero and effectively writes.
//
// With grouped_scores pre-baked by moe_group (scores looked up at the right
// (t, k_slot) per active row), the writer just reads a 32-entry slice of
// grouped_scores per tile-row — no metadata scan, no leids comparison.
ttnn::Tensor moe_ungroup(
    const ttnn::Tensor& expert_out,
    const ttnn::Tensor& plan,
    const ttnn::Tensor& offsets,
    const ttnn::Tensor& grouped_scores,
    uint32_t e_local,
    uint32_t d,
    uint32_t b,
    uint32_t s);

}  // namespace ttml::metal
