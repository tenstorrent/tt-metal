// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tuple>

#include "metal/ttnn_all_includes.hpp"

namespace ttml::metal::ops::moe_group::device {

struct MoeGroupAttributes {
    uint32_t e_local{};
    uint32_t k{};
    uint32_t d{};
    uint32_t b{};
    uint32_t s{};
    uint32_t h{};
    // T_cap upper-bounds total active entries across all local experts
    // plus per-core round_up_align padding plus per-expert tile padding.
    // Computed in moe_group::invoke as:
    //   min(e_local, k) * d * b * s
    //   + e_local * (32 + (l1_align_u32 - 1) * num_total_cores)
    // where l1_align_u32 = L1 alignment / sizeof(uint32_t).
    uint32_t t_cap{};
};

struct MoeGroupTensorArgs {
    ttnn::Tensor dispatched;        // [D, B, S, H]  ROW_MAJOR bf16
    ttnn::Tensor metadata;          // [D, B, S, K]  ROW_MAJOR uint16
    ttnn::Tensor scores;            // [D, B, S, K]  ROW_MAJOR bf16
    ttnn::Tensor local_expert_ids;  // [E_local]      ROW_MAJOR uint16
};

// (grouped, grouped_scores, k_slot, counts, offsets, plan)
//   grouped_scores : [1, 1, 1, T_cap]  ROW_MAJOR bf16   — scores[t, k_slot] per row
//   k_slot         : [1, 1, 1, T_cap]  ROW_MAJOR uint16 — k-slot in metadata[t,:K]
//                                                          per active row
// Both are 0 / 0xFFFF respectively in pad/sentinel slots.
using MoeGroupSpecReturn = std::vector<tt::tt_metal::TensorSpec>;
using MoeGroupTensorReturn =
    std::tuple<ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor, ttnn::Tensor>;

// Aliases required by the ttnn::device_operation framework.
using operation_attributes_t = MoeGroupAttributes;
using tensor_args_t = MoeGroupTensorArgs;
using spec_return_value_t = MoeGroupSpecReturn;
using tensor_return_value_t = MoeGroupTensorReturn;

}  // namespace ttml::metal::ops::moe_group::device
