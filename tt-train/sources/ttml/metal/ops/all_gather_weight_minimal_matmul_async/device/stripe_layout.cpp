// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "stripe_layout.hpp"

#include <algorithm>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>
#include <utility>

#include "kernels/all_gather_weight_minimal_matmul_async_utils.hpp"

namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async {

namespace {

uint32_t stripe_hops_before(uint32_t ring_size, uint32_t ring_index, bool ring) {
    if (!ring) {
        return ring_index;
    }
    return ring_size / 2;
}

uint32_t relay_range(
    const StripeLayout& layout, uint32_t origin, uint32_t c, int32_t dir, bool ring, uint32_t parity_unit) {
    const uint32_t r = layout.ring_size;
    if (!ring) {
        return dir > 0 ? r - 1 - origin : origin;
    }
    const uint32_t half = (r - 1) / 2;
    const bool even = (r % 2) == 0;
    const bool extended = even && ((dir > 0) == ((c / parity_unit) % 2 == 0));
    return extended ? half + 1 : half;
}

}  // namespace

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
    return stripe_regions::total_blocks(ring_size, hops_before, piece_tiles, N_block_tiles);
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
    layout.piece_tiles = tt::div_up(layout.stripe_tiles, num_chains);
    layout.N_block_tiles = N_block_tiles;
    layout.hops_before = stripe_hops_before(ring_size, ring_index, ring);
    layout.stripe_order.push_back(ring_index);
    for (uint32_t dist = 1; dist < stripe_regions::count(ring_size, layout.hops_before); ++dist) {
        if (dist <= layout.hops_before) {
            layout.stripe_order.push_back((ring_index + ring_size - dist) % ring_size);
        }
        if (dist <= stripe_regions::hops_after(ring_size, layout.hops_before)) {
            layout.stripe_order.push_back((ring_index + dist) % ring_size);
        }
    }

    return layout;
}

RelayPlan make_relay_plan(const StripeLayout& layout, uint32_t ring_index, bool ring, uint32_t parity_unit) {
    const uint32_t r = layout.ring_size;
    RelayPlan plan;
    for (const int32_t dir : {1, -1}) {
        auto& segments = dir > 0 ? plan.forward : plan.backward;
        for (uint32_t j = 0; j < r; ++j) {
            const int32_t o = static_cast<int32_t>(ring_index) - (dir * static_cast<int32_t>(j));
            if (!ring && (o < 0 || o >= static_cast<int32_t>(r))) {
                break;
            }
            const uint32_t origin = static_cast<uint32_t>(
                ((o % static_cast<int32_t>(r)) + static_cast<int32_t>(r)) % static_cast<int32_t>(r));
            RelaySegment seg{.stripe = origin, .distance = j, .pieces = {}};
            for (uint32_t c = 0; c < layout.num_chains; ++c) {
                const uint32_t range = relay_range(layout, origin, c, dir, ring, parity_unit);
                if (layout.chain_width(c) == 0 || j + 1 > range) {
                    continue;
                }
                seg.pieces.push_back({c, (j + 2 <= range) ? 1u : 0u});
            }
            if (!seg.pieces.empty()) {
                segments.push_back(std::move(seg));
            }
        }
    }
    return plan;
}
}  // namespace ttml::metal::ops::all_gather_weight_minimal_matmul_async
