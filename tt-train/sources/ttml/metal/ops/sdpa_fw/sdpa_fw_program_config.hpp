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
 *                 them (GQA reuse, more L1); must divide the group size q_heads / kv_heads.
 * max_pieces:     a causal work item heavier than its fair share of a core is cut into up to this many
 *                 pieces along its K/V steps, computed on different cores and merged (1 = never cut,
 *                 at most 63).
 * num_cores:      cores to use (at most the compute grid).
 */
struct SDPAForwardProgramConfig {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
    uint32_t heads_per_item = 0;
    uint32_t max_pieces = 0;
    uint32_t num_cores = 0;
};

}  // namespace ttml::metal
