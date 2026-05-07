// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "internal/risc_attribs.h"

namespace deepseek_b1_ops {

static constexpr uint32_t MAX_SPECULATIVE_TOKENS = 4;
static constexpr uint32_t MAX_WINDOW_TOKENS = MAX_SPECULATIVE_TOKENS + 1;
static constexpr uint32_t TOPK_METADATA_COUNT = 15;
static constexpr uint32_t RELAXED_ACCEPT_TOPN = TOPK_METADATA_COUNT;
static constexpr uint32_t METADATA_PAGE_WORDS = 64;
inline constexpr uint32_t kMetadataTensorBytes = METADATA_PAGE_WORDS * sizeof(uint32_t);

struct DeepseekMetadata {
    // Fixed metadata page layout:
    //   [0] token_type, [1] slot_id, [2] token_id, [3] position_id,
    //   [4] lane_idx, [5] temperature, [6] top_k, [7] top_p,
    //   [8:13] candidate_token_ids, [13:17] prefill_token_ids,
    //   [17:32] p_top15_indices, [32:40] p_top15_scores (two uint16/bf16 scores per word),
    //   [40:55] q_top15_indices, [55:63] q_top15_scores (two uint16/bf16 scores per word).
    uint32_t token_type;
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position_id;
    uint32_t lane_idx;
    float temperature;
    uint32_t top_k;
    float top_p;
    uint32_t candidate_token_ids[MAX_WINDOW_TOKENS];
    uint32_t prefill_token_ids[MAX_SPECULATIVE_TOKENS];
    uint32_t p_top15_indices[TOPK_METADATA_COUNT];
    uint16_t p_top15_scores[TOPK_METADATA_COUNT];
    uint32_t q_top15_indices[TOPK_METADATA_COUNT];
    uint16_t q_top15_scores[TOPK_METADATA_COUNT];
    uint32_t reserved;
};

static_assert(
    sizeof(DeepseekMetadata) == kMetadataTensorBytes, "DeepseekMetadata must stay one 256-byte fixed metadata page");
static_assert(offsetof(DeepseekMetadata, lane_idx) == 16, "lane_idx offset changed");
static_assert(offsetof(DeepseekMetadata, temperature) == 20, "temperature offset changed");
static_assert(offsetof(DeepseekMetadata, top_k) == 24, "top_k offset changed");
static_assert(offsetof(DeepseekMetadata, top_p) == 28, "top_p offset changed");
static_assert(offsetof(DeepseekMetadata, candidate_token_ids) == 32, "candidate_token_ids offset changed");
static_assert(offsetof(DeepseekMetadata, prefill_token_ids) == 52, "prefill_token_ids offset changed");
static_assert(offsetof(DeepseekMetadata, p_top15_indices) == 68, "p_top15_indices offset changed");
static_assert(offsetof(DeepseekMetadata, p_top15_scores) == 128, "p_top15_scores offset changed");
static_assert(offsetof(DeepseekMetadata, q_top15_indices) == 160, "q_top15_indices offset changed");
static_assert(offsetof(DeepseekMetadata, q_top15_scores) == 220, "q_top15_scores offset changed");

}  // namespace deepseek_b1_ops
