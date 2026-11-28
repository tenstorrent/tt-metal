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
#include "tt-train/sources/ttml/metal/ops/all_gather_minimal_matmul_outer_dim_async/device/kernels/all_gather_minimal_matmul_outer_dim_async_utils.hpp"
#include "ttnn/operations/experimental/minimal_matmul/device/kernels/matmul_dataflow_common.hpp"

// Tiles of the gathered operand's outer dimension in one block of this chain: its slice of the stripes, in stripe
// order (kPaddingIndex where the chain is narrower than a piece). Returns false if the whole block is padding.
template <uint32_t block_tiles, uint32_t piece_tiles, uint32_t ring_size, uint32_t hops_before>
inline bool stripe_block_tiles(
    uint32_t block,
    const uint32_t* stripe_order,
    uint32_t stripe_tiles,
    uint32_t chain_offset,
    uint32_t chain_width,
    uint32_t* tiles,
    uint32_t& count) {
    const BlockSpan span = StripeBlocks<block_tiles, piece_tiles, ring_size, hops_before>::span(block);
    count = span.width;
    bool any_real = false;
    for (uint32_t j = 0; j < span.width; j++) {
        // Virtual tile: the chain's pieces of all stripes laid side by side in stripe order.
        const uint32_t v = span.v_start + j;
        const uint32_t local = v % piece_tiles;
        if (local < chain_width) {
            tiles[j] = (stripe_order[v / piece_tiles] * stripe_tiles) + chain_offset + local;
            any_real = true;
        } else {
            tiles[j] = kPaddingIndex;
        }
    }
    return any_real;
}

// Tiles of the local operand's outer dimension in one block: [start, end) of the core's range, kPaddingIndex from the
// tensor's end (`limit`) on. Returns false if the whole block is padding.
inline bool range_block_tiles(uint32_t start, uint32_t end, uint32_t limit, uint32_t* tiles, uint32_t& count) {
    count = end - start;
    for (uint32_t j = 0; j < count; j++) {
        tiles[j] = start + j < limit ? start + j : kPaddingIndex;
    }
    return start < limit;
}

// The rows and columns of the block a core is working on. The gathered operand's side follows the stripe layout and
// changes with the outer loop; the local operand's is a run of the core's own range and changes with the inner loop.
template <
    bool gather_in0,
    uint32_t M_block_tiles,
    uint32_t N_block_tiles,
    uint32_t piece_tiles,
    uint32_t ring_size,
    uint32_t hops_before>
struct BlockTiles {
    static constexpr uint32_t full_height = M_block_tiles;
    static constexpr uint32_t full_width = N_block_tiles;

    uint32_t rows[M_block_tiles];
    uint32_t cols[N_block_tiles];
    uint32_t height = 0;
    uint32_t width = 0;
    bool rows_real = false;  // false: every row of the block is padding
    bool cols_real = false;

    void set_gathered(
        uint32_t block,
        const uint32_t* stripe_order,
        uint32_t stripe_tiles,
        uint32_t chain_offset,
        uint32_t chain_width) {
        if constexpr (gather_in0) {
            rows_real = stripe_block_tiles<M_block_tiles, piece_tiles, ring_size, hops_before>(
                block, stripe_order, stripe_tiles, chain_offset, chain_width, rows, height);
        } else {
            cols_real = stripe_block_tiles<N_block_tiles, piece_tiles, ring_size, hops_before>(
                block, stripe_order, stripe_tiles, chain_offset, chain_width, cols, width);
        }
    }

    void set_local(uint32_t start, uint32_t end, uint32_t limit) {
        if constexpr (gather_in0) {
            cols_real = range_block_tiles(start, end, limit, cols, width);
        } else {
            rows_real = range_block_tiles(start, end, limit, rows, height);
        }
    }
};

// Where an operand's tiles are read from, by their place in its outer dimension: our own stripe comes straight from
// the shard, the others from the persistent buffer the senders fill. A local operand is one stripe that holds it all.
struct StripeSource {
    uint32_t outer_tiles = 0;  // the whole operand
    uint32_t local_first = 0;  // where our own stripe starts
    uint32_t stripe_tiles = 0;
};

// Reads tile (outer, k) of an operand stored [outer, K] (outer_first) or [K, outer].
template <bool outer_first, typename ShardAccessor, typename BufferAccessor>
inline void read_operand_tile(
    const Noc& noc,
    const ShardAccessor& shard,
    const BufferAccessor& buffer,
    const StripeSource& source,
    uint32_t ptr,
    uint32_t tile_bytes,
    uint32_t outer,
    uint32_t k,
    uint32_t K_tiles) {
    if (outer - source.local_first < source.stripe_tiles) {  // inside our own stripe
        noc.async_read(
            shard,
            CoreLocalMem<uint32_t>(ptr),
            tile_bytes,
            {.page_id = operand_page<outer_first>(outer - source.local_first, k, K_tiles, source.stripe_tiles)},
            {});
    } else {
        noc.async_read(
            buffer,
            CoreLocalMem<uint32_t>(ptr),
            tile_bytes,
            {.page_id = operand_page<outer_first>(outer, k, K_tiles, source.outer_tiles)},
            {});
    }
}

// Reads the real rows of an in0 block ([M, K] tiles, a row after the other) for one K block. Padding rows keep what
// the buffer held: their output is never written.
template <uint32_t K_block_tiles, typename ShardAccessor, typename BufferAccessor>
inline void read_in0_block_rows(
    const ShardAccessor& shard,
    const BufferAccessor& buffer,
    const StripeSource& source,
    uint32_t cb_id,
    uint32_t tile_bytes,
    uint32_t K_tiles,
    uint32_t k_start,
    const uint32_t* rows,
    uint32_t height) {
    Noc noc;
    CircularBuffer cb(cb_id);
    const uint32_t base = cb.get_write_ptr();
    for (uint32_t i = 0; i < height; i++) {
        if (rows[i] == kPaddingIndex) {
            continue;
        }
        uint32_t ptr = base + (i * K_block_tiles * tile_bytes);
        for (uint32_t j = 0; j < K_block_tiles; j++, ptr += tile_bytes) {
            const uint32_t k = k_start + j;
            if (k >= K_tiles) {  // K padding
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            } else {
                read_operand_tile<true>(noc, shard, buffer, source, ptr, tile_bytes, rows[i], k, K_tiles);
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

// Reads the real columns of an in1 block ([K, N] tiles, a K row after the other) for one K block.
template <
    uint32_t K_block_tiles,
    uint32_t N_block_tiles,
    bool transpose_b,
    typename ShardAccessor,
    typename BufferAccessor>
inline void read_in1_block_columns(
    const ShardAccessor& shard,
    const BufferAccessor& buffer,
    const StripeSource& source,
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
            if (cols[j] == kPaddingIndex) {
                continue;
            }
            if (k >= K_tiles) {  // K padding
                fill_zeros_async(noc, cb, tile_bytes, ptr - base);
            } else {
                read_operand_tile<transpose_b>(noc, shard, buffer, source, ptr, tile_bytes, cols[j], k, K_tiles);
            }
        }
    }
    noc.async_read_barrier();
    noc.write_zeros_l1_barrier();
}

// Waits until every remote stripe in the block has its first `k_blocks` K blocks in. `tiles` are the block's tiles of
// the gathered dimension. `arrived` remembers what we already saw so each stripe's semaphore is polled only when it
// can still be behind.
inline void wait_for_block_stripes(
    const uint32_t* tiles,
    uint32_t count,
    uint32_t local_stripe,
    uint32_t stripe_tiles,
    const uint32_t* stripe_sem_addrs,
    uint32_t k_blocks,
    uint32_t* arrived) {
    for (uint32_t j = 0; j < count; j++) {
        if (tiles[j] == kPaddingIndex) {
            continue;
        }
        const uint32_t s = tiles[j] / stripe_tiles;
        if (s == local_stripe || arrived[s] >= k_blocks) {
            continue;
        }
        // The sender bumps the low half of this core's semaphore by the K blocks in each unit of our piece of the
        // stripe that has landed - it is written remotely, hence the invalidate.
        const auto* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stripe_sem_addrs[s]);
        do {
            invalidate_l1_cache();
        } while ((*landed & kLandedCountMask) < k_blocks);
        arrived[s] = k_blocks;
    }
}

// Gives back what wait_for_block_stripes counted once the injector is done: `k_blocks` per stripe it waited on.
// Subtracting keeps what a neighbour already delivered for the next call, which a reset would lose.
inline void release_block_stripes(
    const uint32_t* stripe_sem_addrs, const uint32_t* arrived, uint32_t ring_size, uint32_t k_blocks) {
    for (uint32_t s = 0; s < ring_size; s++) {
        if (arrived[s] > 0) {
            noc_semaphore_inc(get_noc_addr(stripe_sem_addrs[s]), -k_blocks);
        }
    }
}

// Writes the real tiles of an output block. granular waits for and frees one row at a time; it's used for the
// last block, where nothing else overlaps the write.
template <bool granular, typename Accessor, typename Block>
inline void write_block_tiles(
    const Accessor& writer, uint32_t N_tiles, uint32_t cb_out_id, uint32_t tile_bytes, const Block& block) {
    Noc noc;
    CircularBuffer cb_out(cb_out_id);
    if constexpr (!granular) {
        cb_out.wait_front(Block::full_height * Block::full_width);
    }
    for (uint32_t i = 0; i < Block::full_height; i++) {
        if constexpr (granular) {
            cb_out.wait_front(Block::full_width);
        }
        if (i < block.height && block.rows[i] != kPaddingIndex) {
            const uint32_t row_ptr =
                granular ? cb_out.get_read_ptr() : cb_out.get_read_ptr() + (i * Block::full_width * tile_bytes);
            const uint32_t row_page = block.rows[i] * N_tiles;
            for (uint32_t j = 0; j < block.width; j++) {
                if (block.cols[j] != kPaddingIndex) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(row_ptr + (j * tile_bytes)),
                        writer,
                        tile_bytes,
                        {},
                        {.page_id = row_page + block.cols[j]});
                }
            }
        }
        if constexpr (granular) {
            noc.async_writes_flushed();
            cb_out.pop_front(Block::full_width);
        }
    }
    if constexpr (!granular) {
        noc.async_writes_flushed();
        cb_out.pop_front(Block::full_height * Block::full_width);
    }
}

// Holds a finished block's output until the writer gets to it during the next block.
template <typename Block>
struct DeferredBlockWrite {
    Block block;
    bool pending = false;

    void defer(const Block& finished) {
        block = finished;
        pending = true;
    }

    template <typename Accessor>
    void write(const Accessor& writer, uint32_t cb_out_id, uint32_t N_tiles, uint32_t tile_bytes) {
        write_block_tiles<false>(writer, N_tiles, cb_out_id, tile_bytes, block);
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
    uint32_t notify = 0;  // the last piece of a K chunk that the next device relays further
};

// Walks the schedule segment -> K chunk -> piece. A K chunk is one or more of the matmul's K blocks, and a unit is one
// piece of one K chunk: the tiles whose arrival the chain's injector on the next device is told about at once.
class SendCursor {
public:
    SendCursor(uint32_t schedule_arg_idx, uint32_t K_tiles, uint32_t chunk_tiles) :
        K_tiles_(K_tiles), chunk_tiles_(chunk_tiles) {
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
    uint32_t chunk_index() const {  // K chunks of the segment in send order; the relay counters count them
        return chunk_idx_;
    }
    bool chunk_begin() const {  // at the K chunk's first piece
        return piece_idx_ == 0;
    }
    uint32_t k_start() const {
        return k_start_;
    }
    uint32_t chunk() const {  // K tiles in this chunk; the last one can be short
        return std::min(chunk_tiles_, K_tiles_ - k_start_);
    }

    // Next piece, then next K chunk, then next segment.
    void next_unit() {
        if (++piece_idx_ < seg_->num_pieces) {
            return;
        }
        piece_idx_ = 0;
        k_start_ += chunk_tiles_;
        ++chunk_idx_;
        if (k_start_ < K_tiles_) {
            return;
        }
        if (segments_left_ == 0) {
            done_ = true;
        } else {
            load_segment();
        }
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
    }

    uint32_t K_tiles_;
    uint32_t chunk_tiles_;
    const tt_l1_ptr uint32_t* args_ = nullptr;
    uint32_t segments_left_ = 0;
    const tt_l1_ptr SendSegment* seg_ = nullptr;
    const tt_l1_ptr SendPieceArgs* pieces_ = nullptr;
    uint32_t chunk_idx_ = 0;
    uint32_t piece_idx_ = 0;
    uint32_t k_start_ = 0;
    bool done_ = true;
};
