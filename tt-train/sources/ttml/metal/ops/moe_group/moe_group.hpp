// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal {

// Groups dispatched MoE tokens by local expert.
// After all_to_all_dispatch, each device holds a dense [D, B, S, H] token
// tensor + a [D, B, S, K] top-K expert-id tensor. This op gathers, for each
// local expert, every token whose top-K mentions it and packs those rows
// into a contiguous tiled tensor so a batched per-expert matmul can run.
//
// Return type: (grouped, grouped_scores, k_slot, counts, offsets, plan)
//   grouped        : [1, 1, T_cap, H]   TILE      bf16   DRAM — active rows
//                                                                packed per expert
//   grouped_scores : [1, 1, 1, T_cap]   ROW_MAJOR bf16   DRAM — scores[t, k_slot]
//                                                                per active row
//                                                                (0 in pad slots)
//   k_slot         : [1, 1, 1, T_cap]   ROW_MAJOR uint16 DRAM — k-slot in
//                                                                metadata[t,:K]
//                                                                per active row
//                                                                (0xFFFF in pad)
//   counts         : [E_local]          ROW_MAJOR uint32 DRAM — real active
//                                                                count per expert
//   offsets        : [E_local + 1]      ROW_MAJOR uint32 DRAM — per-expert row
//                                                                prefix-sum (each
//                                                                expert rounded
//                                                                up to a 32-row
//                                                                tile boundary)
//   plan           : [T_cap]            ROW_MAJOR uint32 DRAM — per-grouped-row
//                                                                source index in
//                                                                dispatched
//                                                                (SENTINEL =
//                                                                0xFFFFFFFF for
//                                                                pad slots)
//
// T_cap is the upper bound on the number of rows written. Host must allocate
// for the worst case because actual routing is only known after phase 1:
//
//   T_cap = round_up_32(min(E_local, K) · T_total
//                       + E_local · (32 + (cursor_align-1) · N))
//                       └──── worst-case active + padding ────┘
//
//   T_total      = D · B · S        (total dispatched tokens)
//   N            = num_total_cores  (worker grid size, typically 72 on WH Galaxy)
//   cursor_align = L1_ALIGN_BYTES / 2  (= 8 on WH/BH; bumped from L1_ALIGN_U32=4
//                                       so per-core writes of bf16 grouped_scores
//                                       and uint16 k_slot land on 16 B boundaries
//                                       alongside uint32 plan)
//
// - Worst-case active: each of T_total tokens picks K experts, at most
//   min(E_local, K) of those are local.
// - Padding: up to 32 rows per expert for per-expert tile alignment in
//   grouped, plus up to (cursor_align-1)·N SENTINEL rows per expert to keep
//   per-core write boundaries 16 B-aligned for NOC L1→DRAM writes of all
//   three side tensors (plan, grouped_scores, k_slot). The final capacity is
//   also rounded to a 32-row tile boundary because downstream kernels consume
//   plan/grouped_scores/expert_out in 32-row chunks.
using MoeGroupResult = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

MoeGroupResult moe_group(
    const ttnn::Tensor& dispatched,        // [D, B, S, H]  ROW_MAJOR bf16
    const ttnn::Tensor& metadata,          // [D, B, S, K]  ROW_MAJOR uint16
    const ttnn::Tensor& scores,            // [D, B, S, K]  ROW_MAJOR bf16
    const ttnn::Tensor& local_expert_ids,  // [E_local]     ROW_MAJOR uint16
    uint32_t e_local,
    uint32_t k);

}  // namespace ttml::metal
