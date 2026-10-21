// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttml::metal {

/**
 * Tunables for the SDPA forward kernel. Both are sequence lengths in elements and must be
 * multiples of TILE_HEIGHT (32). 0 leaves the choice to the program factory.
 *
 * - q_chunk_size: query rows handled together against one K/V stream. Every K/V tile a core
 *   reads from DRAM feeds q_chunk_size/32 query tile-rows instead of one, so DRAM traffic for
 *   K and V drops by that factor. Must divide the sequence length. Costs L1 (attention
 *   weights, running stats and output accumulators all scale with it) and, for causal
 *   attention, a little wasted work above the diagonal inside the chunk.
 *
 * - k_chunk_size: K/V rows consumed per inner-loop step. The online-softmax correction
 *   (rescale of the running sum and output) runs once per step, so larger values amortize
 *   it. Must divide the sequence length. Costs L1 for the K, V and score buffers.
 *
 * Work is distributed over Q chunks. For causal masks, chunks are paired first-with-last,
 * second-with-second-to-last and so on, so every pair costs the same number of K/V steps.
 */
struct SDPAForwardProgramConfig {
    uint32_t q_chunk_size = 0;
    uint32_t k_chunk_size = 0;
};

}  // namespace ttml::metal
