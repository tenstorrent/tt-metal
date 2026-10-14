// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

constexpr uint32_t kPaddingColumn = 0xFFFFFFFFu;

struct BlockSpan {
    uint32_t v_start = 0;
    uint32_t width = 0;
};

template <uint32_t N_block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t subblock_w>
struct StripeBlocks {
    static constexpr uint32_t local_blocks = (piece_tiles + N_block_tiles - 1) / N_block_tiles;
    static constexpr uint32_t remote_tiles = (ring_size - 1) * piece_tiles;
    static constexpr uint32_t remote_blocks = (remote_tiles + N_block_tiles - 1) / N_block_tiles;
    static constexpr uint32_t num_blocks = local_blocks + remote_blocks;

    static BlockSpan balanced(uint32_t b, uint32_t tiles, uint32_t blocks) {
        const uint32_t units = (tiles + subblock_w - 1) / subblock_w;
        const uint32_t base = units / blocks;
        const uint32_t rem = units % blocks;
        const uint32_t v_start = subblock_w * ((b * base) + std::min(b, rem));
        const uint32_t width = subblock_w * (base + (b < rem ? 1u : 0u));
        return {v_start, std::min(width, tiles - v_start)};
    }

    static BlockSpan span(uint32_t n_block_iter) {
        if (n_block_iter < local_blocks) {
            return balanced(n_block_iter, piece_tiles, local_blocks);
        }
        const BlockSpan r = balanced(n_block_iter - local_blocks, remote_tiles, remote_blocks);
        return {piece_tiles + r.v_start, r.width};
    }
};
