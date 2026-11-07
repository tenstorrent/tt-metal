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
#include "tt-train/sources/ttml/metal/ops/all_gather_weight_minimal_matmul_async/device/kernels/all_gather_weight_minimal_matmul_async_utils.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

// Global N columns of one N block for this chain (kPaddingColumn where the chain is narrower than a piece).
// Returns false if the whole block is padding.
template <uint32_t N_block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t hops_before>
inline bool block_columns(
    uint32_t n_block_iter,
    const uint32_t* stripe_order,
    uint32_t stripe_tiles,
    uint32_t chain_offset,
    uint32_t chain_width,
    uint32_t* cols,
    uint32_t& width) {
    const BlockSpan span = StripeBlocks<N_block_tiles, piece_tiles, ring_size, hops_before>::span(n_block_iter);
    width = span.width;
    bool any_real = false;
    for (uint32_t j = 0; j < span.width; j++) {
        // Virtual column: the chain's pieces of all stripes laid side by side in stripe order.
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

// Our own stripe comes straight from the shard, the others from the persistent buffer the senders fill.
template <
    uint32_t K_block_tiles,
    uint32_t N_block_tiles,
    bool transpose_b,
    typename ShardAccessor,
    typename PwbAccessor>
inline void read_in1_block_columns(
    const ShardAccessor& shard,
    const PwbAccessor& pwb,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t K_tiles,
    uint32_t N_tiles,
    uint32_t k_start,
    const uint32_t* cols,
    uint32_t width,
    uint32_t local_stripe,
    uint32_t stripe_tiles) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t base = cb.get_write_ptr();
    const uint32_t local_first = local_stripe * stripe_tiles;
    for (uint32_t i = 0; i < K_block_tiles; i++) {
        const uint32_t k = k_start + i;
        uint32_t ptr = base + (i * N_block_tiles * tile_bytes);
        for (uint32_t j = 0; j < width; j++, ptr += tile_bytes) {
            if (cols[j] == kPaddingColumn) {
                continue;
            }
            if (k >= K_tiles) {  // K padding
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            } else if (cols[j] - local_first < stripe_tiles) {  // inside our own stripe
                noc.async_read(
                    shard,
                    CoreLocalMem<uint32_t>(ptr),
                    tile_bytes,
                    {.page_id = weight_page<transpose_b>(cols[j] - local_first, k, K_tiles, stripe_tiles)},
                    {});
            } else {
                noc.async_read(
                    pwb,
                    CoreLocalMem<uint32_t>(ptr),
                    tile_bytes,
                    {.page_id = weight_page<transpose_b>(cols[j], k, K_tiles, N_tiles)},
                    {});
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

// Waits until every remote stripe in the block has its first `chunks` K chunks in. `arrived` remembers what we
// already saw so each stripe's semaphore is polled only when it can still be behind.
inline void wait_for_block_stripes(
    const uint32_t* cols,
    uint32_t width,
    uint32_t local_stripe,
    uint32_t stripe_tiles,
    const uint32_t* stripe_sem_addrs,
    uint32_t chunks,
    uint32_t* arrived) {
    for (uint32_t j = 0; j < width; j++) {
        if (cols[j] == kPaddingColumn) {
            continue;
        }
        const uint32_t s = cols[j] / stripe_tiles;
        if (s == local_stripe || arrived[s] >= chunks) {
            continue;
        }
        // The sender bumps this core's semaphore once per K chunk of our piece of the stripe.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[s]), chunks);
        arrived[s] = chunks;
    }
}

// Writes the real columns of an output block. granular waits for and frees one row at a time; it's used for the
// last block, where nothing else overlaps the write.
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

// Holds a finished block's output until the writer gets to it during the next block.
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

// The sender's schedule as the factory writes it into the runtime args.
struct SendSegment {
    uint32_t stripe = 0;
    uint32_t distance = 0;  // hops the stripe already travelled to reach us; 0 is our own
    uint32_t num_pieces = 0;
};

struct SendPieceArgs {
    uint32_t row_offset = 0;  // the chain's slice of the stripe
    uint32_t width = 0;
    uint32_t inj_noc_x = 0;  // the chain's injector, which waits for this piece on the next device
    uint32_t inj_noc_y = 0;
    uint32_t notify = 0;  // the next device relays this piece further
};

// Walks segment -> K chunk -> piece -> packet; a unit is one piece of one K chunk.
class SendCursor {
public:
    SendCursor(uint32_t schedule_arg_idx, uint32_t K_tiles, uint32_t chunk_tiles, uint32_t tiles_per_packet) :
        K_tiles_(K_tiles), chunk_tiles_(chunk_tiles), tiles_per_packet_(tiles_per_packet) {
        args_ = reinterpret_cast<const tt_l1_ptr uint32_t*>(get_arg_addr(schedule_arg_idx));
        segments_left_ = *args_++;
        done_ = segments_left_ == 0;
        if (!done_) {
            load_segment();
        }
    }

    bool done() const {
        return done_;
    }
    const SendSegment& segment() const {
        return *seg_;
    }
    const SendPieceArgs& piece() const {
        return pieces_[piece_idx_];
    }
    // Units of the segment in send order; the relay counters count in the same order.
    uint32_t unit_index() const {
        return (chunk_idx_ * seg_->num_pieces) + piece_idx_;
    }
    uint32_t k_start() const {
        return k_start_;
    }
    uint32_t chunk() const {  // K tiles in this chunk; the last one can be short
        return chunk_;
    }
    uint32_t row() const {
        return row_;
    }
    uint32_t k() const {
        return k_start_ + k_in_chunk_;
    }
    uint32_t count() const {  // tiles in the current packet; the unit's last packet can be short
        return std::min(tiles_per_packet_, unit_tiles_ - t0_);
    }
    bool unit_begin() const {  // are we at the first packet of a unit?
        return t0_ == 0;
    }
    bool unit_end() const {  // are we at the last packet of a unit?
        return t0_ + tiles_per_packet_ >= unit_tiles_;
    }

    // Next packet. Past the end of a unit: next piece, then next K chunk, then next segment.
    void advance() {
        if (!unit_end()) {
            t0_ += tiles_per_packet_;
            k_in_chunk_ += tiles_per_packet_;
            while (k_in_chunk_ >= chunk_) {
                k_in_chunk_ -= chunk_;
                ++row_;
            }
            return;
        }
        t0_ = 0;
        if (++piece_idx_ == seg_->num_pieces) {
            piece_idx_ = 0;
            k_start_ += chunk_tiles_;
            ++chunk_idx_;
            if (k_start_ >= K_tiles_) {
                if (segments_left_ == 0) {
                    done_ = true;
                } else {
                    load_segment();
                }
                return;
            }
        }
        start_unit();
    }

private:
    // Segments sit back to back in the runtime args: the header, then its pieces.
    void load_segment() {
        seg_ = reinterpret_cast<const tt_l1_ptr SendSegment*>(args_);
        pieces_ = reinterpret_cast<const tt_l1_ptr SendPieceArgs*>(seg_ + 1);
        args_ = reinterpret_cast<const tt_l1_ptr uint32_t*>(pieces_ + seg_->num_pieces);
        --segments_left_;
        k_start_ = 0;
        chunk_idx_ = 0;
        piece_idx_ = 0;
        t0_ = 0;
        start_unit();
    }

    void start_unit() {
        chunk_ = std::min(chunk_tiles_, K_tiles_ - k_start_);
        unit_tiles_ = pieces_[piece_idx_].width * chunk_;
        row_ = 0;
        k_in_chunk_ = 0;
    }

    uint32_t K_tiles_;
    uint32_t chunk_tiles_;
    uint32_t tiles_per_packet_;
    const tt_l1_ptr uint32_t* args_ = nullptr;
    uint32_t segments_left_ = 0;
    const tt_l1_ptr SendSegment* seg_ = nullptr;
    const tt_l1_ptr SendPieceArgs* pieces_ = nullptr;
    uint32_t chunk_idx_ = 0;
    uint32_t piece_idx_ = 0;
    uint32_t k_start_ = 0;
    uint32_t chunk_ = 0;
    uint32_t t0_ = 0;  // first tile of the current packet within the unit
    uint32_t row_ = 0;
    uint32_t k_in_chunk_ = 0;
    uint32_t unit_tiles_ = 0;
    bool done_ = true;
};
