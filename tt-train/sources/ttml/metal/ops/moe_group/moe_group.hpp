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
// Return type: (grouped, counts, offsets, plan)
//   grouped : [1, 1, T_cap, H]  TILE bf16  DRAM — active rows packed per expert
//   counts  : [E_local]          uint32     DRAM — real active count per expert
//   offsets : [E_local + 1]      uint32     DRAM — per-expert row prefix-sum
//                                                   (each expert rounded up
//                                                    to a 32-row tile boundary)
//   plan    : [T_cap]            uint32     DRAM — per-grouped-row source index
//                                                   in dispatched (SENTINEL =
//                                                   0xFFFFFFFF for pad slots)
//
// T_cap is the upper bound on the number of rows written to grouped / plan.
// Host must allocate for the worst case because actual routing is only known
// after phase 1 on the device:
//
//   T_cap = min(E_local, K) · T_total  +  E_local · (32 + 3 · N)
//           └── worst-case active ──┘     └─── padding ───┘
//
//   T_total = D · B · S        (total dispatched tokens)
//   N       = num_total_cores  (worker grid size, typically 72 on WH Galaxy)
//
// - Worst-case active: each of T_total tokens picks K experts, at most
//   min(E_local, K) of those are local.
// - Padding: up to 32 rows per expert for per-expert tile alignment in
//   grouped, plus up to 3·N SENTINEL rows per expert to keep per-core plan
//   write boundaries 16B-aligned for NOC L1→DRAM writes.
using MoeGroupResult = std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

MoeGroupResult moe_group(
    const ttnn::Tensor& dispatched,        // [D, B, S, H]  ROW_MAJOR bf16
    const ttnn::Tensor& metadata,          // [D, B, S, K]  ROW_MAJOR int32
    const ttnn::Tensor& local_expert_ids,  // [E_local]      ROW_MAJOR uint32
    uint32_t e_local,
    uint32_t k);

}  // namespace ttml::metal
