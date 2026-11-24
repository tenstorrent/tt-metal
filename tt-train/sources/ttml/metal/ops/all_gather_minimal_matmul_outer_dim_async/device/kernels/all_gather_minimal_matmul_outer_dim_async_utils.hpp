// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

// A block row or column that holds no tile: it pads a narrow chain up to piece_tiles, or a core's range past the end
// of the tensor.
constexpr uint32_t kPaddingIndex = 0xFFFFFFFFu;

// Every core has one semaphore per stripe. Its low half counts the K blocks of the stripe that landed for the core's
// injector and its high half the K chunks that landed for the core's fabric sender to relay, so a core can be both.
// Each half holds at most a call's K blocks plus those a neighbour already delivered for the next call.
constexpr uint32_t kRelayCountShift = 16;
constexpr uint32_t kLandedCountMask = (1U << kRelayCountShift) - 1;

// send_on_grid.cpp's compile-time args before the route and the tensor accessors.
constexpr uint32_t kSendFixedCtArgs = 10;

// Page of tile (outer, k) of a matmul operand, where outer is its row (in0) or its column (in1): the operand is stored
// [outer, K] (outer_first) or [K, outer].
template <bool outer_first>
inline uint32_t operand_page(uint32_t outer, uint32_t k, uint32_t K_tiles, uint32_t outer_tiles) {
    if constexpr (outer_first) {
        return (outer * K_tiles) + k;
    } else {
        return (k * outer_tiles) + outer;
    }
}

struct BlockSpan {
    uint32_t v_start = 0;
    uint32_t width = 0;
};

// Region r holds the pieces at hop distance r. The gathered dimension's blocks (N blocks of in1, M blocks of in0) are
// cut per region, so a block never waits on two hop distances.
namespace stripe_regions {
constexpr uint32_t hops_after(uint32_t ring_size, uint32_t hops_before) {
    return ring_size - 1 - hops_before;
}
constexpr uint32_t count(uint32_t ring_size, uint32_t hops_before) {
    return 1 + std::max(hops_before, hops_after(ring_size, hops_before));
}
constexpr uint32_t pieces(uint32_t r, uint32_t ring_size, uint32_t hops_before) {
    return r == 0 ? 1u : (r <= hops_before ? 1u : 0u) + (r <= hops_after(ring_size, hops_before) ? 1u : 0u);
}
constexpr uint32_t blocks(
    uint32_t r, uint32_t ring_size, uint32_t hops_before, uint32_t piece_tiles, uint32_t block_tiles) {
    return ((pieces(r, ring_size, hops_before) * piece_tiles) + block_tiles - 1) / block_tiles;
}
constexpr uint32_t total_blocks(uint32_t ring_size, uint32_t hops_before, uint32_t piece_tiles, uint32_t block_tiles) {
    uint32_t n = 0;
    for (uint32_t r = 0; r < count(ring_size, hops_before); r++) {
        n += blocks(r, ring_size, hops_before, piece_tiles, block_tiles);
    }
    return n;
}
}  // namespace stripe_regions

template <uint32_t block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t hops_before>
struct StripeBlocks {
    static constexpr uint32_t num_regions = stripe_regions::count(ring_size, hops_before);
    static constexpr uint32_t region_pieces(uint32_t r) {
        return stripe_regions::pieces(r, ring_size, hops_before);
    }
    static constexpr uint32_t region_blocks(uint32_t r) {
        return stripe_regions::blocks(r, ring_size, hops_before, piece_tiles, block_tiles);
    }
    static constexpr uint32_t num_blocks =
        stripe_regions::total_blocks(ring_size, hops_before, piece_tiles, block_tiles);

    // Where a block starts (in virtual tiles: the chain's pieces of all stripes side by side) and how wide it is. Each
    // region is split into ceil(tiles / block_tiles) blocks of near-equal width, so no block crosses a region or ends
    // as a tiny tail.
    static BlockSpan span(uint32_t block) {
        uint32_t first_piece = 0;
        for (uint32_t r = 0; r < num_regions; r++) {
            const uint32_t nb = region_blocks(r);
            if (block < nb) {
                const uint32_t tiles = region_pieces(r) * piece_tiles;
                const uint32_t base = tiles / nb;
                const uint32_t rem = tiles % nb;
                return {
                    (first_piece * piece_tiles) + (block * base) + std::min(block, rem),
                    base + (block < rem ? 1u : 0u)};
            }
            block -= nb;
            first_piece += region_pieces(r);
        }
        return {0, 0};
    }
};
