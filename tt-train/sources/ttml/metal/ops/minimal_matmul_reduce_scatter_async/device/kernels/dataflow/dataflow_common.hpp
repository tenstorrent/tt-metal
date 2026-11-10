// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "tt-train/sources/ttml/metal/ops/minimal_matmul_reduce_scatter_async/device/kernels/minimal_matmul_reduce_scatter_async_utils.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

struct CoreShare {
    uint32_t m_off = 0;
    uint32_t m_width = 0;
    uint32_t n_off = 0;
    uint32_t n_width = 0;
};

struct BlockRegion {
    uint32_t row0 = 0;
    uint32_t rows = 0;
    uint32_t col0 = 0;
    uint32_t cols = 0;
    bool empty() const {
        return rows == 0 || cols == 0;
    }
};

template <uint32_t slice_tiles, uint32_t m_piece, uint32_t M_blocks, uint32_t N_blocks>
inline BlockRegion block_region(uint32_t slice, uint32_t mb, uint32_t nb, const CoreShare& share) {
    const mmrs::Span m = mmrs::clip(mmrs::balanced_span(mb, m_piece, M_blocks), share.m_width);
    const mmrs::Span n = mmrs::n_block(nb, share.n_width, N_blocks);
    return {(slice * slice_tiles) + share.m_off + m.start, m.size, share.n_off + n.start, n.size};
}

template <uint32_t K_block_tiles, bool transpose_a, typename Accessor>
inline void read_in0_block(
    const Accessor& input,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t M_tiles,
    uint32_t K_tiles,
    uint32_t k_start,
    uint32_t row0,
    uint32_t rows) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t m = 0; m < rows; m++) {
        uint32_t ptr = base + (m * K_block_tiles * tile_bytes);
        for (uint32_t i = 0; i < K_block_tiles; i++, ptr += tile_bytes) {
            const uint32_t k = k_start + i;
            if (k >= K_tiles) {
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            } else {
                noc.async_read(
                    input,
                    CoreLocalMem<uint32_t>(ptr),
                    tile_bytes,
                    {.page_id = mmrs::input_page<transpose_a>(row0 + m, k, M_tiles, K_tiles)},
                    {});
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

template <uint32_t K_block_tiles, uint32_t N_block_tiles, bool transpose_b, typename Accessor>
inline void read_in1_block(
    const Accessor& weight,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t K_tiles,
    uint32_t N_tiles,
    uint32_t k_start,
    uint32_t col0,
    uint32_t cols) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t i = 0; i < K_block_tiles; i++) {
        const uint32_t k = k_start + i;
        uint32_t ptr = base + (i * N_block_tiles * tile_bytes);
        for (uint32_t j = 0; j < cols; j++, ptr += tile_bytes) {
            if (k >= K_tiles) {
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            } else {
                noc.async_read(
                    weight,
                    CoreLocalMem<uint32_t>(ptr),
                    tile_bytes,
                    {.page_id = mmrs::weight_page<transpose_b>(col0 + j, k, K_tiles, N_tiles)},
                    {});
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

template <uint32_t M_block_tiles, uint32_t N_block_tiles, bool granular, typename Accessor>
inline void write_block(
    const Accessor& dst,
    uint32_t cb_out_id,
    uint32_t tile_bytes,
    uint32_t N_tiles,
    uint32_t first_row,
    const BlockRegion& region) {
    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    if constexpr (!granular) {
        cb_out.wait_front(M_block_tiles * N_block_tiles);
    }
    for (uint32_t m = 0; m < M_block_tiles; m++) {
        if constexpr (granular) {
            cb_out.wait_front(N_block_tiles);
        }
        if (m < region.rows) {
            const uint32_t row_ptr =
                granular ? cb_out.get_read_ptr() : cb_out.get_read_ptr() + (m * N_block_tiles * tile_bytes);
            const uint32_t page = ((region.row0 + m - first_row) * N_tiles) + region.col0;
            for (uint32_t j = 0; j < region.cols; j++) {
                noc.async_write(
                    CoreLocalMem<uint32_t>(row_ptr + (j * tile_bytes)), dst, tile_bytes, {}, {.page_id = page + j});
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

template <uint32_t M_block_tiles, uint32_t N_block_tiles, typename Accessor>
inline void read_incoming_block(
    const Accessor& buffer, uint32_t cb_id, uint32_t tile_bytes, uint32_t N_tiles, const BlockRegion& region) {
    Noc noc;
    CircularBuffer cb(cb_id);
    cb.reserve_back(M_block_tiles * N_block_tiles);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t m = 0; m < region.rows; m++) {
        const uint32_t page = ((region.row0 + m) * N_tiles) + region.col0;
        uint32_t ptr = base + (m * N_block_tiles * tile_bytes);
        for (uint32_t j = 0; j < region.cols; j++, ptr += tile_bytes) {
            noc.async_read(buffer, CoreLocalMem<uint32_t>(ptr), tile_bytes, {.page_id = page + j}, {});
        }
    }
    noc.async_read_barrier();
    cb.push_back(M_block_tiles * N_block_tiles);
}

struct RingWriterArgs {
    uint32_t ring_index = 0;
    uint32_t defer_write_k_block = 0;
    uint32_t landed_sem_addr = 0;
    uint32_t sender_noc_x = 0;
    uint32_t sender_noc_y = 0;
    uint32_t go_sem_addr = 0;
};

template <
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t ring_size,
    uint32_t slice_tiles,
    typename OutAccessor,
    typename BufferAccessor>
class RingWriter {
public:
    RingWriter(
        const OutAccessor& out,
        const BufferAccessor& buffer,
        uint32_t tile_bytes,
        uint32_t N_tiles,
        const RingWriterArgs& args) :
        out_(out), buffer_(buffer), tile_bytes_(tile_bytes), N_tiles_(N_tiles), args_(args) {
    }

    void at_k_block(uint32_t k_block) {
        signal_ready();
        if (deferred_.pending && k_block == args_.defer_write_k_block) {
            write_deferred();
        }
    }

    void fetch_incoming(uint32_t step, const BlockRegion& region) {
        signal_ready();
        if (step == 0) {
            return;
        }
        if (!region.empty()) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args_.landed_sem_addr), ++consumed_);
        }
        read_incoming_block<M_block_tiles, N_block_tiles>(buffer_, kIncomingCb, tile_bytes_, N_tiles_, region);
    }

    void end_block(uint32_t step, const BlockRegion& region, bool last) {
        if (last) {
            write_block<M_block_tiles, N_block_tiles, true>(
                out_, kOutCb, tile_bytes_, N_tiles_, own_first_row(), region);
            return;
        }
        deferred_ = {.region = region, .staged = step + 1 < ring_size, .pending = true};
    }

    void finish() {
        if (deferred_.pending) {
            write_deferred();
        }
        Noc noc;
        noc.async_write_barrier();
        signal_ready();
        if (consumed_ > 0) {
            noc_semaphore_inc(get_noc_addr(args_.landed_sem_addr), -consumed_);
        }
    }

private:
    static constexpr uint32_t kOutCb = tt::CBIndex::c_2;
    static constexpr uint32_t kIncomingCb = tt::CBIndex::c_5;

    struct Deferred {
        BlockRegion region;
        bool staged = false;
        bool pending = false;
    };

    uint32_t own_first_row() const {
        return args_.ring_index * slice_tiles;
    }

    void write_deferred() {
        if (deferred_.staged) {
            write_block<M_block_tiles, N_block_tiles, false>(
                buffer_, kOutCb, tile_bytes_, N_tiles_, 0, deferred_.region);
            unsignalled_++;
        } else {
            write_block<M_block_tiles, N_block_tiles, false>(
                out_, kOutCb, tile_bytes_, N_tiles_, own_first_row(), deferred_.region);
        }
        deferred_.pending = false;
    }

    void signal_ready() {
        if (unsignalled_ == 0) {
            return;
        }
        Noc noc;
        noc.async_write_barrier();
        if (ready_addr_ == 0) {
            auto* go = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(args_.go_sem_addr);
            noc_semaphore_wait_min(go, 1);
            ready_addr_ = *go;
        }
        noc_semaphore_inc(get_noc_addr(args_.sender_noc_x, args_.sender_noc_y, ready_addr_), unsignalled_);
        unsignalled_ = 0;
    }

    const OutAccessor& out_;
    const BufferAccessor& buffer_;
    uint32_t tile_bytes_;
    uint32_t N_tiles_;
    RingWriterArgs args_;
    Deferred deferred_;
    uint32_t consumed_ = 0;
    uint32_t unsignalled_ = 0;
    uint32_t ready_addr_ = 0;
};
