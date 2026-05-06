// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/risc_attribs.h"

namespace deepseek_b1_ops {

static constexpr uint32_t MAX_SPECULATIVE_TOKENS = 4;
static constexpr uint32_t MAX_WINDOW_TOKENS = MAX_SPECULATIVE_TOKENS + 1;
static constexpr uint32_t RELAXED_ACCEPT_TOPN = 10;
static constexpr uint32_t METADATA_PAGE_WORDS = 64;

struct DeepseekMetadata {
    // Fixed metadata page layout:
    //   [0] token_type, [1] slot_id, [2] token_id, [3] position_id, [4] prefill_token_id,
    //   [5] lane_idx, [6] window_start_pos, [7] num_window_tokens,
    //   [8:13] candidate_token_ids, [13:18] candidate_positions,
    //   [18] target_topn_count, [19:29] target_topn_tokens, [29:39] target_topn_probs.
    uint32_t token_type;
    uint32_t slot_id;
    uint32_t token_id;
    uint32_t position_id;
    uint32_t prefill_token_id;
    uint32_t lane_idx;
    uint32_t window_start_pos;
    uint32_t num_window_tokens;
    uint32_t candidate_token_ids[MAX_WINDOW_TOKENS];
    uint32_t candidate_positions[MAX_WINDOW_TOKENS];
    uint32_t target_topn_count;
    uint32_t target_topn_tokens[RELAXED_ACCEPT_TOPN];
    uint32_t target_topn_probs[RELAXED_ACCEPT_TOPN];
    uint32_t reserved[25];
};

static_assert(sizeof(DeepseekMetadata) == 256, "DeepseekMetadata must stay one 256-byte fixed metadata page");

}  // namespace deepseek_b1_ops
