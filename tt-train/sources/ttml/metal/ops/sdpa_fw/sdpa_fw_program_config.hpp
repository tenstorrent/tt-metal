// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

/**
 * SDPA forward tunables; 0 = let the program factory pick. Reachable from metal::sdpa_fw only (the
 * benchmark and the tests): the autograd op and ring SDPA run the factory's pick.
 * q_chunk_size:   query rows that share one pass over K/V (fewer K/V reads, more L1); elements, a
 *                 multiple of 32 that divides S.
 * k_chunk_size:   K/V rows per inner-loop step (fewer softmax corrections, more L1); same rule.
 * heads_per_item: query heads of one K/V group processed together, so one K/V read serves all of
 *                 them (GQA reuse, more L1); must divide the group size q_heads / kv_heads. The pick
 *                 is 1: more rows per item only pay off with cost-based scheduling.
 */
struct SDPAForwardProgramConfig {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
    uint32_t heads_per_item = 0;
};

}  // namespace ttml::metal
