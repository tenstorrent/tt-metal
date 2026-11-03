// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

/**
 * SDPA forward tunables; 0 = let the program factory pick. Reachable from metal::sdpa_fw only (the
 * benchmark and the tests): the autograd op runs the factory's pick, ring SDPA pins main's program
 * (ring_program_config in ring_sdpa_fw_program_factory.cpp).
 * q_chunk_size:   query rows that share one pass over K/V (fewer K/V reads, more L1); elements, a
 *                 multiple of 32 that divides S.
 * k_chunk_size:   K/V rows per inner-loop step (fewer softmax corrections, more L1); same rule.
 * heads_per_item: query heads of one K/V group processed together, so one K/V (and mask) read serves
 *                 all of them (GQA reuse, more L1); must divide the group size q_heads / kv_heads. The
 *                 pick stacks up to four heads with a mask tensor, where the shared mask read pays, and
 *                 one otherwise (without a mask tensor the stacking measured flat).
 * max_pieces:     a causal work item heavier than its fair share of a core is cut into up to this many
 *                 pieces along its K/V steps, computed on different cores and merged (0 = 4, 1 = never
 *                 cut, at most 63). Above 1 the L1 check also reserves the partials' circular buffers.
 *                 A program that cuts owns a DRAM scratch for the pieces' partials, allocated on the
 *                 program-cache miss and held for the life of the cache entry, not of the call:
 *                 slots * item rows * (d_v / 32 + 2) fp32 tiles, about 5 to 12 MiB at d = 128 with the
 *                 default picks (at most one slot per core, scaling with the rows of an item). Every
 *                 distinct shape that cuts holds its own, so a workload with many sequence lengths
 *                 (padded rollouts, say) should set 1 unless the cut shapes are few.
 * num_cores:      cores to use (at most the compute grid).
 * causal_pairs:   place the causal work items as main did instead of longest first: query chunk i of a
 *                 stack with chunk (n - 1 - i), the pairs in contiguous blocks over the cores (when the
 *                 chunk count is even and the pairs cover the cores; contiguous blocks of items
 *                 otherwise). Never cuts, so max_pieces must be 0 or 1. Ring SDPA sets it, with main's
 *                 chunk sizes, so its chunking and core assignment are the ones it ran before the knobs.
 *
 * TODO: this runs parallel to ttnn's SDPAProgramConfig (ttnn/operations/transformer/sdpa_config.hpp),
 * which already carries q_chunk_size / k_chunk_size, and the causal light/heavy pairing in
 * metal/common/sdpa_work_split.hpp repeats sdpa_program_factory.cpp's. Converge on SDPAProgramConfig
 * (heads_per_item with 0 = auto, the same 0-means-auto rule for the chunk sizes) so tt-train and ttnn
 * callers tune SDPA the same way; tracked in https://github.com/tenstorrent/tt-metal/issues/63774 since
 * it changes ttnn's public config.
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
