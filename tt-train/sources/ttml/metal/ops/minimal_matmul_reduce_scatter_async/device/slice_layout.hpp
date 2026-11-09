// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "kernels/minimal_matmul_reduce_scatter_async_utils.hpp"

namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async {
struct SliceLayout {
    uint32_t ring_size = 1;
    uint32_t slice_tiles = 0;
    uint32_t m_cores = 1;
    uint32_t n_cores = 1;
    uint32_t N_tiles = 0;
    uint32_t m_piece = 0;
    uint32_t M_blocks = 0;
    uint32_t n_piece = 0;
    uint32_t N_blocks = 0;
    uint32_t n_first = 0;

    mmrs::Span m_share(uint32_t m_core) const;
    mmrs::Span n_share(uint32_t n_core) const;
    mmrs::Span m_block(uint32_t mb) const;
    mmrs::Span n_block(uint32_t n_core, uint32_t nb) const;
};

SliceLayout make_slice_layout(
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t ring_size,
    uint32_t m_cores,
    uint32_t n_cores,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    bool narrow_first_n_share = false);

uint64_t padded_work_per_core(
    uint32_t M_tiles, uint32_t N_tiles, uint32_t ring_size, uint32_t m_cores, uint32_t n_cores);

}  // namespace ttml::metal::ops::minimal_matmul_reduce_scatter_async
