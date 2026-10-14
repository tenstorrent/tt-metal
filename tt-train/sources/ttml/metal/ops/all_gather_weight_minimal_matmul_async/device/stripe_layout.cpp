// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stripe_layout.hpp"

#include <algorithm>
#include <tt_stl/assert.hpp>

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {
uint32_t StripeLayout::chain_offset(uint32_t c) const {
    const uint32_t base = stripe_tiles / num_chains;
    const uint32_t rem = stripe_tiles % num_chains;
    return (c * base) + std::min(c, rem);
}

uint32_t StripeLayout::chain_width(uint32_t c) const {
    const uint32_t base = stripe_tiles / num_chains;
    const uint32_t rem = stripe_tiles % num_chains;
    return base + (c < rem ? 1u : 0u);
}

uint32_t StripeLayout::n_blocks_per_core() const {
    return local_blocks + remote_blocks;
}

std::vector<uint32_t> stripe_consumption_order(uint32_t ring_size, uint32_t ring_index, bool ring) {
    std::vector<uint32_t> order;
    order.reserve(ring_size);
    order.push_back(ring_index);
    std::vector<bool> taken(ring_size, false);
    taken[ring_index] = true;
    const int32_t r = static_cast<int32_t>(ring_size);
    const int32_t d = static_cast<int32_t>(ring_index);
    for (int32_t dist = 1; static_cast<uint32_t>(order.size()) < ring_size; ++dist) {
        for (const int32_t s : {d - dist, d + dist}) {
            int32_t idx = s;
            if (ring) {
                idx = ((s % r) + r) % r;
            } else if (s < 0 || s >= r) {
                continue;
            }
            if (!taken[idx]) {
                taken[idx] = true;
                order.push_back(static_cast<uint32_t>(idx));
            }
        }
    }
    return order;
}

StripeLayout make_stripe_layout(
    uint32_t N_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t num_chains, uint32_t N_block_tiles, bool ring) {
    TT_FATAL(ring_size >= 1, "ring_size must be >= 1, got {}", ring_size);
    TT_FATAL(ring_index < ring_size, "ring_index ({}) must be < ring_size ({})", ring_index, ring_size);
    TT_FATAL(
        N_tiles % ring_size == 0,
        "N_tiles ({}) must be divisible by ring_size ({}): each device must own a tile-aligned stripe",
        N_tiles,
        ring_size);
    TT_FATAL(
        num_chains >= 1 && N_block_tiles >= 1,
        "num_chains ({}) and N_block_tiles ({}) must be >= 1.",
        num_chains,
        N_block_tiles);

    StripeLayout layout;
    layout.ring_size = ring_size;
    layout.stripe_tiles = N_tiles / ring_size;
    layout.num_chains = num_chains;
    layout.piece_tiles = (layout.stripe_tiles + num_chains - 1) / num_chains;
    layout.N_block_tiles = N_block_tiles;
    layout.local_blocks = (layout.piece_tiles + N_block_tiles - 1) / N_block_tiles;
    layout.remote_blocks = (((ring_size - 1) * layout.piece_tiles) + N_block_tiles - 1) / N_block_tiles;
    layout.stripe_order = stripe_consumption_order(ring_size, ring_index, ring);

    return layout;
}
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
