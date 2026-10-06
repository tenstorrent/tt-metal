// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_tail, rope core reader (one core pair per lane): the -1 scalar tile, the query cos/sin tiles (the first pair),
// then the hand-offs from its norm core: the query's RoPE tiles once (the first pair), and per lane of this pair the
// pooled key's RoPE tiles (after signalling room for them) with block-start cos/sin tiles whose row 0 is the lane's
// row.
// Compile-time args: TensorAccessorArgs cos, sin, block_cos, block_sin.
// Runtime args: 0 cos, 1 sin, 2 block_cos, 3 block_sin addresses, 4 rows, 5 peer x, 6 peer y, 7 lane_first,
// 8 lane_count, 9 do_query.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "index_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace index_tail;

void kernel_main() {
    const uint32_t cos_addr = get_arg_val<uint32_t>(0);
    const uint32_t sin_addr = get_arg_val<uint32_t>(1);
    const uint32_t bcos_addr = get_arg_val<uint32_t>(2);
    const uint32_t bsin_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t peer_x = get_arg_val<uint32_t>(5);
    const uint32_t peer_y = get_arg_val<uint32_t>(6);
    const uint32_t lane_first = get_arg_val<uint32_t>(7);
    const uint32_t lane_count = get_arg_val<uint32_t>(8);
    const uint32_t do_query = get_arg_val<uint32_t>(9);
    constexpr auto cos_args = TensorAccessorArgs<0>();
    constexpr auto sin_args = TensorAccessorArgs<cos_args.next_compile_time_args_offset()>();
    constexpr auto bcos_args = TensorAccessorArgs<sin_args.next_compile_time_args_offset()>();
    constexpr auto bsin_args = TensorAccessorArgs<bcos_args.next_compile_time_args_offset()>();
    const auto cos = TensorAccessor(cos_args, cos_addr);
    const auto sin = TensorAccessor(sin_args, sin_addr);
    const auto bcos = TensorAccessor(bcos_args, bcos_addr);
    const auto bsin = TensorAccessor(bsin_args, bsin_addr);
    volatile tt_l1_ptr uint32_t* sem_q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_Q));
    volatile tt_l1_ptr uint32_t* sem_k = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_K));
    const uint64_t peer_ready = get_noc_addr(peer_x, peer_y, get_semaphore(SEM_READY));

    generate_bcast_col_scalar(CircularBuffer(CB_SCALAR), 0xBF800000u);
    if (do_query) {
        FUSED_ZONE("fz_qs_it_rr_query");
        cb_reserve_back(CB_COS, ROPE_TILES);
        cb_reserve_back(CB_SIN, ROPE_TILES);
        for (uint32_t t = 0; t < ROPE_TILES; ++t) {
            noc_async_read_page(t, cos, get_write_ptr(CB_COS) + t * TILE_BYTES);
            noc_async_read_page(t, sin, get_write_ptr(CB_SIN) + t * TILE_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_COS, ROPE_TILES);
        cb_push_back(CB_SIN, ROPE_TILES);

        cb_reserve_back(CB_IN_Q, ROPE_TILES);
        cb_reserve_back(CB_ROT_Q, ROPE_TILES);
        noc_semaphore_wait_min(sem_q, 1);
        cb_push_back(CB_IN_Q, ROPE_TILES);
        cb_push_back(CB_ROT_Q, ROPE_TILES);
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_it_rr_lane");
        const uint32_t lane = lane_first + i;
        cb_reserve_back(CB_IN_K, ROPE_TILES);
        cb_reserve_back(CB_ROT_K, ROPE_TILES);
        noc_semaphore_inc(peer_ready, 1);
        cb_reserve_back(CB_BCOS, ROPE_TILES);
        cb_reserve_back(CB_BSIN, ROPE_TILES);
        const uint32_t bcos_l1 = get_write_ptr(CB_BCOS), bsin_l1 = get_write_ptr(CB_BSIN);
        for (uint32_t t = 0; t < ROPE_TILES; ++t) {
            noc_async_read_page(t, bcos, bcos_l1 + t * TILE_BYTES);
            noc_async_read_page(t, bsin, bsin_l1 + t * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        if (lane != 0) {  // the pooled row of the lane is row 0 of its tiles: give it the lane's block-start row
            for (uint32_t t = 0; t < ROPE_TILES; ++t) {
                copy_tile_row(bcos_l1 + t * TILE_BYTES, lane, bcos_l1 + t * TILE_BYTES, 0, false);
                copy_tile_row(bsin_l1 + t * TILE_BYTES, lane, bsin_l1 + t * TILE_BYTES, 0, false);
            }
        }
        cb_push_back(CB_BCOS, ROPE_TILES);
        cb_push_back(CB_BSIN, ROPE_TILES);
        noc_semaphore_wait_min(sem_k, i + 1);
        cb_push_back(CB_IN_K, ROPE_TILES);
        cb_push_back(CB_ROT_K, ROPE_TILES);
    }
    noc_semaphore_set(sem_q, 0);
    noc_semaphore_set(sem_k, 0);
    noc_async_atomic_barrier();
}
