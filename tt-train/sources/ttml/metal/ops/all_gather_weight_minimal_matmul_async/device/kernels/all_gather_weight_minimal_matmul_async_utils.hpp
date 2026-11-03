// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

constexpr uint32_t kPaddingColumn = 0xFFFFFFFFu;

template <bool transpose_b>
inline uint32_t weight_page(uint32_t n, uint32_t k, uint32_t K_tiles, uint32_t n_cols) {
    if constexpr (transpose_b) {
        return (n * K_tiles) + k;
    } else {
        return (k * n_cols) + n;
    }
}

struct BlockSpan {
    uint32_t v_start = 0;
    uint32_t width = 0;
};

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
    uint32_t r, uint32_t ring_size, uint32_t hops_before, uint32_t piece_tiles, uint32_t N_block_tiles) {
    return ((pieces(r, ring_size, hops_before) * piece_tiles) + N_block_tiles - 1) / N_block_tiles;
}
constexpr uint32_t total_blocks(
    uint32_t ring_size, uint32_t hops_before, uint32_t piece_tiles, uint32_t N_block_tiles) {
    uint32_t n = 0;
    for (uint32_t r = 0; r < count(ring_size, hops_before); r++) {
        n += blocks(r, ring_size, hops_before, piece_tiles, N_block_tiles);
    }
    return n;
}
}  // namespace stripe_regions

template <uint32_t N_block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t hops_before>
struct StripeBlocks {
    static constexpr uint32_t num_regions = stripe_regions::count(ring_size, hops_before);
    static constexpr uint32_t region_pieces(uint32_t r) {
        return stripe_regions::pieces(r, ring_size, hops_before);
    }
    static constexpr uint32_t region_blocks(uint32_t r) {
        return stripe_regions::blocks(r, ring_size, hops_before, piece_tiles, N_block_tiles);
    }
    static constexpr uint32_t num_blocks =
        stripe_regions::total_blocks(ring_size, hops_before, piece_tiles, N_block_tiles);

    static BlockSpan span(uint32_t n_block_iter) {
        uint32_t first_piece = 0;
        for (uint32_t r = 0; r < num_regions; r++) {
            const uint32_t nb = region_blocks(r);
            if (n_block_iter < nb) {
                const uint32_t tiles = region_pieces(r) * piece_tiles;
                const uint32_t base = tiles / nb;
                const uint32_t rem = tiles % nb;
                return {
                    (first_piece * piece_tiles) + (n_block_iter * base) + std::min(n_block_iter, rem),
                    base + (n_block_iter < rem ? 1u : 0u)};
            }
            n_block_iter -= nb;
            first_piece += region_pieces(r);
        }
        return {0, 0};
    }
};
