// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, norm core writer (one core pair per lane): the first pair hands the normalized query's RoPE tiles (0, 1
// and swapped) to its rope core and writes tiles 2, 3 of the index query; per lane of this core drains the canonical
// ring to DRAM, hands the pooled key's RoPE tiles over once the rope core has room, and writes row 0 of the pooled
// key's tiles 2, 3 into the compressed cache row (P // 4) of the lane.
// Compile-time args: TensorAccessorArgs index_query, ring, cache, kv_block_start, kv_row_hit.
// Runtime args: 0 index_query, 1 ring, 2 cache, 3 kv_block_start, 4 kv_row_hit addresses, 5 rows, 6 lane tile rows, 7
// peer x, 8 peer y, 9 lane_first, 10 lane_count, 11 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "index_tail_cbs.h"
#include "positions.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t ring_addr = get_arg_val<uint32_t>(1);
    const uint32_t cache_addr = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t hit_addr = get_arg_val<uint32_t>(4);
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
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto out = TensorAccessor(out_args, out_addr);
    const auto ring = TensorAccessor(ring_args, ring_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);
    volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_READY));
    const uint64_t peer_sem_q = get_noc_addr(peer_x, peer_y, get_semaphore(SEM_Q));
    const uint64_t peer_sem_k = get_noc_addr(peer_x, peer_y, get_semaphore(SEM_K));
    // the peer's CBs sit at the same L1 addresses (every CB is declared on both cores and these are unused here)
    const uint64_t peer_in_q = get_noc_addr(peer_x, peer_y, get_write_ptr(CB_IN_Q));
    const uint64_t peer_rot_q = get_noc_addr(peer_x, peer_y, get_write_ptr(CB_ROT_Q));
    const uint64_t peer_in_k = get_noc_addr(peer_x, peer_y, get_write_ptr(CB_IN_K));
    const uint64_t peer_rot_k = get_noc_addr(peer_x, peer_y, get_write_ptr(CB_ROT_K));

    cb_reserve_back(CB_POS_W, 1);
    cb_reserve_back(CB_POSCW, 1);
    const uint32_t pos_l1 = get_write_ptr(CB_POS_W);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCW));
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);

    if (do_query) {
        cb_wait_front(CB_NQ, HEAD_TILES);
        const uint32_t l1 = get_read_ptr(CB_NQ);
        noc_async_write(l1, peer_in_q, ROPE_TILES * TILE_BYTES);
        noc_async_write(l1 + TILE_BYTES, peer_rot_q, TILE_BYTES);
        noc_async_write(l1, peer_rot_q + TILE_BYTES, TILE_BYTES);
        for (uint32_t c = ROPE_TILES; c < HEAD_TILES; ++c) {
            noc_async_write_page(c, out, l1 + c * TILE_BYTES);
        }
        noc_async_write_barrier();
        noc_semaphore_inc(peer_sem_q, 1);
        cb_pop_front(CB_NQ, HEAD_TILES);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        const uint32_t lane = lane_first + i;
        cb_wait_front(CB_RINGW, HEAD_TILES);
        {
            const uint32_t l1 = get_read_ptr(CB_RINGW);
            for (uint32_t c = 0; c < HEAD_TILES; ++c) {
                noc_async_write_page(lane * HEAD_TILES + c, ring, l1 + c * TILE_BYTES);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(CB_RINGW, HEAD_TILES);

        cb_wait_front(CB_NK, HEAD_TILES);
        {
            const uint32_t l1 = get_read_ptr(CB_NK);
            noc_semaphore_wait_min(ready, i + 1);
            noc_async_write(l1, peer_in_k, ROPE_TILES * TILE_BYTES);
            noc_async_write(l1 + TILE_BYTES, peer_rot_k, TILE_BYTES);
            noc_async_write(l1, peer_rot_k + TILE_BYTES, TILE_BYTES);
            noc_async_write_barrier();
            noc_semaphore_inc(peer_sem_k, 1);
            const uint32_t row = positions[lane] >> 2;
            const uint32_t tile_row = lane * lane_tile_rows + (row >> 5);
            for (uint32_t c = ROPE_TILES; c < HEAD_TILES; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        l1 + c * TILE_BYTES + chunk_offset(0, half),
                        cache.get_noc_addr(tile_row * HEAD_TILES + c, chunk_offset(row & 31, half)),
                        ROW_BYTES);
                }
            }
            noc_async_write_barrier();
        }
        cb_pop_front(CB_NK, HEAD_TILES);
    }
    noc_semaphore_set(ready, 0);
    noc_async_atomic_barrier();
}
