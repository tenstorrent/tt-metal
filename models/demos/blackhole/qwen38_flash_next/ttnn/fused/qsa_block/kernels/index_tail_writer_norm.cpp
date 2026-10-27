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
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "index_tail_cbs.h"
#include "positions.h"

using namespace index_tail;

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
    volatile tt_l1_ptr uint32_t* ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(ready_semaphore_id));
    const uint64_t peer_sem_q = get_noc_addr(peer_x, peer_y, get_semaphore(query_semaphore_id));
    const uint64_t peer_sem_k = get_noc_addr(peer_x, peer_y, get_semaphore(key_semaphore_id));
    // the peer's CBs sit at the same L1 addresses (every CB is declared on both cores and these are unused here)
    const uint64_t peer_in_q = get_noc_addr(peer_x, peer_y, get_write_ptr(cb_in_q));
    const uint64_t peer_rot_q = get_noc_addr(peer_x, peer_y, get_write_ptr(cb_rot_q));
    const uint64_t peer_in_k = get_noc_addr(peer_x, peer_y, get_write_ptr(cb_in_k));
    const uint64_t peer_rot_k = get_noc_addr(peer_x, peer_y, get_write_ptr(cb_rot_k));

    CircularBuffer(cb_pos_w).reserve_back(1);
    CircularBuffer(cb_poscw).reserve_back(1);
    const uint32_t positions_l1 = get_write_ptr(cb_pos_w);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1),
        block_start,
        row_hit,
        rows,
        get_write_ptr(cb_poscw));
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(positions_l1);

    if (do_query) {
        CircularBuffer(cb_nq).wait_front(head_tiles);
        const uint32_t l1 = get_read_ptr(cb_nq);
        noc_async_write(l1, peer_in_q, rope_tiles * tile_bytes);
        noc_async_write(l1 + tile_bytes, peer_rot_q, tile_bytes);
        noc_async_write(l1, peer_rot_q + tile_bytes, tile_bytes);
        for (uint32_t c = rope_tiles; c < head_tiles; ++c) {
            noc_async_write_page(c, out, l1 + c * tile_bytes);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(peer_sem_q, 1);
        CircularBuffer(cb_nq).pop_front(head_tiles);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        CircularBuffer(cb_ringw).wait_front(head_tiles);
        {
            const uint32_t l1 = get_read_ptr(cb_ringw);
            for (uint32_t c = 0; c < head_tiles; ++c) {
                noc_async_write_page(lane * head_tiles + c, ring, l1 + c * tile_bytes);
            }
            noc_async_write_barrier();
        }
        CircularBuffer(cb_ringw).pop_front(head_tiles);

        CircularBuffer(cb_nk).wait_front(head_tiles);
        {
            const uint32_t l1 = get_read_ptr(cb_nk);
            noc_semaphore_wait_min(ready, i + 1);
            noc_async_write(l1, peer_in_k, rope_tiles * tile_bytes);
            noc_async_write(l1 + tile_bytes, peer_rot_k, tile_bytes);
            noc_async_write(l1, peer_rot_k + tile_bytes, tile_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(peer_sem_k, 1);
            const uint32_t row = positions[lane] >> 2;
            const uint32_t tile_row = lane * lane_tile_rows + (row >> 5);
            for (uint32_t c = rope_tiles; c < head_tiles; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        l1 + c * tile_bytes + chunk_offset(0, half),
                        cache.get_noc_addr(tile_row * head_tiles + c, chunk_offset(row & 31, half)),
                        half_row_bytes);
                }
            }
            noc_async_write_barrier();
        }
        CircularBuffer(cb_nk).pop_front(head_tiles);
    }
    noc_semaphore_set(ready, 0);
    noc_async_atomic_barrier();
}
