// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "tt-train/sources/ttml/metal/ops/all_gather_weight_minimal_matmul_async/device/kernels/all_gather_weight_minimal_matmul_async_utils.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

template <uint32_t N_block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t subblock_w>
inline bool block_columns(
    uint32_t n_block_iter,
    const uint32_t* stripe_order,
    uint32_t stripe_tiles,
    uint32_t chain_offset,
    uint32_t chain_width,
    uint32_t* cols,
    uint32_t& width) {
    const BlockSpan span = StripeBlocks<N_block_tiles, piece_tiles, ring_size, subblock_w>::span(n_block_iter);
    width = span.width;
    bool any_real = false;
    for (uint32_t j = 0; j < span.width; j++) {
        const uint32_t v = span.v_start + j;
        const uint32_t local = v % piece_tiles;
        if (local < chain_width) {
            cols[j] = (stripe_order[v / piece_tiles] * stripe_tiles) + chain_offset + local;
            any_real = true;
        } else {
            cols[j] = kPaddingColumn;
        }
    }
    return any_real;
}

template <uint32_t K_block_tiles, uint32_t N_block_tiles, typename Accessor>
inline void read_in1_block_columns(
    const Accessor& reader,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t K_tiles,
    uint32_t k_start,
    const uint32_t* cols,
    uint32_t width) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t i = 0; i < K_block_tiles; i++) {
        const uint32_t k = k_start + i;
        uint32_t ptr = base + (i * N_block_tiles * tile_bytes);
        for (uint32_t j = 0; j < width; j++, ptr += tile_bytes) {
            if (cols[j] == kPaddingColumn) {
                continue;
            }
            if (k < K_tiles) {
                noc.async_read(
                    reader, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = (cols[j] * K_tiles) + k}, {});
            } else {
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool granular, typename Accessor>
inline void write_block_columns(
    const Accessor& writer,
    uint32_t M_tiles,
    uint32_t N_tiles,
    uint32_t cb_out_id,
    uint32_t tile_bytes,
    uint32_t m_start,
    uint32_t m_end,
    const uint32_t* cols,
    uint32_t width) {
    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    if constexpr (!granular) {
        cb_out.wait_front(M_block_tiles * N_block_tiles);
    }
    for (uint32_t m_id = 0; m_id < M_block_tiles; m_id++) {
        if constexpr (granular) {
            cb_out.wait_front(N_block_tiles);
        }
        const uint32_t m = m_start + m_id;
        if (m < m_end && m < M_tiles) {
            const uint32_t row_ptr =
                granular ? cb_out.get_read_ptr() : cb_out.get_read_ptr() + (m_id * N_block_tiles * tile_bytes);
            for (uint32_t j = 0; j < width; j++) {
                if (cols[j] != kPaddingColumn) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(row_ptr + (j * tile_bytes)),
                        writer,
                        tile_bytes,
                        {},
                        {.page_id = (m * N_tiles) + cols[j]});
                }
            }
        }
        if constexpr (granular) {
            noc.async_writes_flushed();
            cb_out.pop_front(N_block_tiles);
        }
    }
    if constexpr (!granular) {
        noc.async_writes_flushed();
        cb_out.pop_front(M_block_tiles * N_block_tiles);
    }
}

template <uint32_t M_block_tiles, uint32_t N_block_tiles>
struct DeferredBlockWrite {
    uint32_t cols[N_block_tiles];
    uint32_t width = 0;
    uint32_t m_start = 0;
    uint32_t m_end = 0;
    bool pending = false;

    void defer(uint32_t block_m_start, uint32_t block_m_end, const uint32_t* block_cols, uint32_t block_width) {
        m_start = block_m_start;
        m_end = block_m_end;
        width = block_width;
        for (uint32_t j = 0; j < block_width; j++) {
            cols[j] = block_cols[j];
        }
        pending = true;
    }

    template <typename Accessor>
    void write(const Accessor& writer, uint32_t cb_out_id, uint32_t M_tiles, uint32_t N_tiles, uint32_t tile_bytes) {
        write_block_columns<M_block_tiles, N_block_tiles, false>(
            writer, M_tiles, N_tiles, cb_out_id, tile_bytes, m_start, m_end, cols, width);
        pending = false;
    }
};
