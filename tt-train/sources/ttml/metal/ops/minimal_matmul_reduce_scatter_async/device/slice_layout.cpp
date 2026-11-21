// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_layout.hpp"

#include <algorithm>
#include <tt-metalium/math.hpp>
#include <tt_stl/assert.hpp>

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
mmrs::Span SliceLayout::m_share(uint32_t m_core) const {
    return mmrs::balanced_span(m_core, slice_tiles, m_cores);
}

mmrs::Span SliceLayout::n_share(uint32_t n_core) const {
    if (n_core == 0) {
        return {0, n_first};
    }
    const mmrs::Span rest = mmrs::balanced_span(n_core - 1, N_tiles - n_first, n_cores - 1);
    return {n_first + rest.start, rest.size};
}

mmrs::Span SliceLayout::m_block(uint32_t mb) const {
    return mmrs::balanced_span(mb, m_piece, M_blocks);
}

mmrs::Span SliceLayout::n_block(uint32_t n_core, uint32_t nb) const {
    return mmrs::n_block(nb, n_share(n_core).size, N_blocks);
}

uint32_t SliceLayout::blocks_per_step(uint32_t dir) const {
    return mmrs::blocks_in_direction(N_blocks, num_dirs, dir) * M_blocks;
}

SliceLayout make_slice_layout(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t ring_size,
    uint32_t m_cores,
    uint32_t n_cores,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t num_dirs,
    bool narrow_first_n_share) {
    TT_FATAL(ring_size >= 1, "ring_size must be >= 1, got {}", ring_size);
    TT_FATAL(
        M_tiles % ring_size == 0,
        "M_tiles ({}) must be divisible by ring_size ({}): every device keeps a tile-aligned slice of rows",
        M_tiles,
        ring_size);
    TT_FATAL(M_tiles > 0 && N_tiles > 0, "M_tiles ({}) and N_tiles ({}) must be > 0", M_tiles, N_tiles);
    TT_FATAL(m_cores >= 1 && n_cores >= 1, "m_cores ({}) and n_cores ({}) must be >= 1", m_cores, n_cores);
    TT_FATAL(
        M_block_tiles >= 1 && N_block_tiles >= 1,
        "M_block_tiles ({}) and N_block_tiles ({}) must be >= 1",
        M_block_tiles,
        N_block_tiles);
    TT_FATAL(num_dirs == 1 || num_dirs == 2, "num_dirs must be 1 or 2, got {}", num_dirs);

    SliceLayout layout;
    layout.ring_size = ring_size;
    layout.slice_tiles = M_tiles / ring_size;
    layout.m_cores = m_cores;
    layout.n_cores = n_cores;
    layout.N_tiles = N_tiles;
    layout.m_piece = tt::div_up(layout.slice_tiles, m_cores);
    layout.M_blocks = tt::div_up(layout.m_piece, M_block_tiles);
    layout.n_piece = tt::div_up(N_tiles, n_cores);
    layout.N_blocks = tt::div_up(layout.n_piece, N_block_tiles);
    layout.num_dirs = num_dirs;
    layout.n_first = mmrs::balanced_span(0, N_tiles, n_cores).size;
    if (!narrow_first_n_share || n_cores < 2) {
        return layout;
    }
    uint32_t blocks = layout.N_blocks;
    uint32_t piece = tt::div_up(N_tiles + blocks, n_cores);
    if (tt::div_up(piece, N_block_tiles) != blocks) {
        blocks = tt::div_up(piece, N_block_tiles);
        piece = tt::div_up(N_tiles + blocks, n_cores);
    }
    if (piece < 2 * blocks) {
        return layout;
    }
    layout.n_first = piece - blocks;
    layout.n_piece = std::max(layout.n_first, tt::div_up(N_tiles - layout.n_first, n_cores - 1));
    layout.N_blocks = tt::div_up(layout.n_piece, N_block_tiles);
    return layout;
}

uint64_t padded_work_per_core(
    uint32_t M_tiles, uint32_t N_tiles, uint32_t ring_size, uint32_t m_cores, uint32_t n_cores) {
    return static_cast<uint64_t>(ring_size) * tt::div_up(M_tiles / ring_size, m_cores) * tt::div_up(N_tiles, n_cores);
}

std::vector<std::vector<uint32_t>> assign_producers(uint32_t num_cores, uint32_t num_senders) {
    TT_FATAL(num_senders >= 1, "num_senders must be >= 1, got {}", num_senders);
    std::vector<std::vector<uint32_t>> producers(num_senders);
    for (uint32_t core = 0; core < num_cores; ++core) {
        producers[core % num_senders].push_back(core);
    }
    return producers;
}
}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async
