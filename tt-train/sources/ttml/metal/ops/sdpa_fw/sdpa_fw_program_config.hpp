// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

/**
 * SDPA forward tunables; 0 = let the program factory pick. Reachable from metal::sdpa_fw only (the
 * benchmark and the tests): the autograd op runs the factory's pick, ring SDPA pins main's program
 * (ring_program_config).
 * q_chunk_size:   query rows that share one pass over K/V (fewer K/V reads, more L1); elements, a
 *                 multiple of 32 that divides S.
 * k_chunk_size:   K/V rows per inner-loop step (fewer softmax corrections, more L1); same rule.
 * heads_per_item: query heads of one K/V group processed together, so one K/V read serves all of
 *                 them (GQA reuse, more L1); must divide the group size q_heads / kv_heads.
 * max_pieces:     a causal work item heavier than its fair share of a core is cut into up to this many
 *                 pieces along its K/V steps, computed on different cores and merged (0 = 4, 1 = never
 *                 cut, at most 63). Above 1 the L1 check also reserves the partials' circular buffers.
 * num_cores:      cores to use (at most the compute grid).
 * causal_pairs:   place the causal work items as main did instead of longest first: query chunk i of a
 *                 stack with chunk (n - 1 - i), the pairs in contiguous blocks over the cores (when the
 *                 chunk count is even and the pairs cover the cores; contiguous blocks of items
 *                 otherwise). Never cuts, so max_pieces must be 0 or 1. Ring SDPA sets it, with main's
 *                 chunk sizes, so its chunking and core assignment are the ones it ran before the knobs.
 */
struct SDPAForwardProgramConfig {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
    uint32_t heads_per_item = 0;
    uint32_t max_pieces = 0;
    uint32_t num_cores = 0;
    bool causal_pairs = false;
};

}  // namespace ttml::metal
