// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core writer: the first pair hands the normalized query's RoPE tiles (0, 1 and swapped) to its rope
// core and writes tiles 2, 3 of the index query; per lane drains the ring to DRAM, hands the pooled key's RoPE tiles
// over once the rope core has room, and writes row 0 of the pooled key's tiles 2, 3 into compressed cache row P // 4.
// Compile-time args: TensorAccessorArgs index_query, ring, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 index_query, 1 ring, 2 cache, 3 kv_block_start, 4 kv_row_hit addresses, 5 rows, 6 lane tile rows, 7
// peer x, 8 peer y, 9 lane_first, 10 lane_count, 11 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "index_tail_cbs.h"
#include "positions.h"

using namespace index_tail;

namespace {
// The front tiles 0, 1 of `normed_cb` to the peer's in CB (in order) and rot CB (swapped: tile 1, then tile 0), at
// `peer_in` / `peer_rot` on core (peer_x, peer_y); the caller barriers.
FORCE_INLINE void hand_off_rope_tiles(
    uint32_t normed_cb, uint32_t peer_x, uint32_t peer_y, uint32_t peer_in, uint32_t peer_rot) {
    const Noc noc;
    noc.async_write(
        CircularBuffer(normed_cb),
        UnicastEndpoint{},
        rope_tiles * tile_bytes,
        {},
        {.noc_x = peer_x, .noc_y = peer_y, .addr = peer_in});
    noc.async_write(
        CircularBuffer(normed_cb),
        UnicastEndpoint{},
        tile_bytes,
        {.offset_bytes = tile_bytes},
        {.noc_x = peer_x, .noc_y = peer_y, .addr = peer_rot});
    noc.async_write(
        CircularBuffer(normed_cb),
        UnicastEndpoint{},
        tile_bytes,
        {},
        {.noc_x = peer_x, .noc_y = peer_y, .addr = peer_rot + tile_bytes});
}
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t cache_addr = get_arg_val<uint32_t>(2);
    const uint32_t block_start_addr = get_arg_val<uint32_t>(3);
    const uint32_t row_hit_addr = get_arg_val<uint32_t>(4);
    const uint32_t rows = get_arg_val<uint32_t>(5);
    const uint32_t lane_tile_rows = get_arg_val<uint32_t>(6);
    const uint32_t peer_x = get_arg_val<uint32_t>(7);
    const uint32_t peer_y = get_arg_val<uint32_t>(8);
    const uint32_t lane_first = get_arg_val<uint32_t>(9);
    const uint32_t lane_count = get_arg_val<uint32_t>(10);
    const uint32_t do_query = get_arg_val<uint32_t>(11);
    constexpr auto out_args = TensorAccessorArgs<0>();
    constexpr auto ring_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr auto cache_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr auto block_start_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto row_hit_args = TensorAccessorArgs<block_start_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto ring = TensorAccessor(ring_args, ring_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto block_start = TensorAccessor(block_start_args, block_start_addr);
    const auto row_hit = TensorAccessor(row_hit_args, row_hit_addr);
    const Noc noc;
    // this core's room counter, and the rope core's (the peer's) query / key hand-off counters
    Semaphore<> rope_ready(ready_semaphore_id), query_handoff(query_semaphore_id), key_handoff(key_semaphore_id);
    // the peer's CBs sit at the same L1 addresses (every CB is declared on both cores and these are unused here)
    const uint32_t peer_in_q = CircularBuffer(cb_in_q).get_write_ptr();
    const uint32_t peer_rot_q = CircularBuffer(cb_rot_q).get_write_ptr();
    const uint32_t peer_in_k = CircularBuffer(cb_in_k).get_write_ptr();
    const uint32_t peer_rot_k = CircularBuffer(cb_rot_k).get_write_ptr();

    CircularBuffer(cb_pos_w).reserve_back(1);
    CircularBuffer(cb_poscw).reserve_back(1);
    // cb_pos_w is scratch: reserved once and never pushed
    const CoreLocalMem<volatile uint32_t> positions(CircularBuffer(cb_pos_w).get_write_ptr());
    tile_rows::read_positions(positions, block_start, row_hit, rows, CircularBuffer(cb_poscw).get_write_ptr());

    if (do_query) {
        CircularBuffer(cb_nq).wait_front(head_tiles);
        hand_off_rope_tiles(cb_nq, peer_x, peer_y, peer_in_q, peer_rot_q);
        for (uint32_t c = rope_tiles; c < head_tiles; ++c) {
            noc.async_write(
                CircularBuffer(cb_nq),
                out,
                out.get_aligned_page_size(),
                {.offset_bytes = c * tile_bytes},
                {.page_id = c});
        }
        noc.async_write_barrier();
        query_handoff.up(noc, peer_x, peer_y, 1);
        CircularBuffer(cb_nq).pop_front(head_tiles);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_ringw).wait_front(head_tiles);
        {
            for (uint32_t c = 0; c < head_tiles; ++c) {
                noc.async_write(
                    CircularBuffer(cb_ringw),
                    ring,
                    ring.get_aligned_page_size(),
                    {.offset_bytes = c * tile_bytes},
                    {.page_id = lane * head_tiles + c});
            }
            noc.async_write_barrier();
        }
        CircularBuffer(cb_ringw).pop_front(head_tiles);

        CircularBuffer(cb_nk).wait_front(head_tiles);
        {
            rope_ready.wait_min(i + 1);
            hand_off_rope_tiles(cb_nk, peer_x, peer_y, peer_in_k, peer_rot_k);
            noc.async_write_barrier();
            key_handoff.up(noc, peer_x, peer_y, 1);
            const uint32_t row = positions[lane] >> 2;
            const uint32_t tile_row = lane * lane_tile_rows + (row >> 5);
            for (uint32_t c = rope_tiles; c < head_tiles; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_write(
                        CircularBuffer(cb_nk),
                        cache,
                        half_row_bytes,
                        {.offset_bytes = c * tile_bytes + chunk_offset(0, half)},
                        {.page_id = tile_row * head_tiles + c, .offset_bytes = chunk_offset(row & 31, half)});
                }
            }
            noc.async_write_barrier();
        }
        CircularBuffer(cb_nk).pop_front(head_tiles);
    }
    rope_ready.set(0);
    noc.async_atomic_barrier();
}
