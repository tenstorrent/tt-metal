// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core reader (one staging core per lane): the ones tile, the positions, the value tiles into pack
// tiles 0..7 (the key's rotated tiles 8, 9 and normalized tiles 10..15 arrive from the key cores); for this core's
// lanes (lane_first .. lane_first + lane_count - 1) the lane's KV staging tiles with row (P % 32) replaced by the
// lane's packed row [v | k] (-0 -> +0 as the chain's SFPU one-hot select does).
// Compile-time args: TensorAccessorArgs staging, v, kv_block_start, kv_row_hit.  Runtime args: 0 staging, 1 v, 2
// kv_block_start, 3 kv_row_hit, 4 rows, 5 first tile of the value in v (0 for the separate projection shard; the v
// window's first tile in the merged projection shard), 6 lane_first, 7 lane_count (this staging core's lanes).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "positions.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t stg_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t v_first = get_arg_val<uint32_t>(5);
    const uint32_t lane_first = get_arg_val<uint32_t>(6);
    const uint32_t lane_count = get_arg_val<uint32_t>(7);
    constexpr auto stg_args = TensorAccessorArgs<0>();
    constexpr auto v_args = TensorAccessorArgs<stg_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto stg = TensorAccessor(stg_args, stg_addr);
    const auto v = TensorAccessor(v_args, v_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);
    Semaphore<> sem_krot(SEM_KROT), sem_knorm(SEM_KNORM);

    cb_reserve_back(CB_ONES, 1);
    tile_rows::fill_words(get_write_ptr(CB_ONES), TILE_BYTES / 4, 0x3F803F80u);
    cb_push_back(CB_ONES, 1);
    cb_reserve_back(CB_POS_R, 1);
    cb_reserve_back(CB_POSCR, 1);
    cb_reserve_back(CB_PACK, PACK_TILES);
    const uint32_t pos_l1 = get_write_ptr(CB_POS_R), pack_l1 = get_write_ptr(CB_PACK);
    tile_rows::read_positions(
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCR));
    {
        FUSED_ZONE("fz_qs_mt_rs_value");
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(v_first + c, v, pack_l1 + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    volatile tt_l1_ptr uint32_t* positions = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1);
    {
        FUSED_ZONE("fz_qs_mt_rs_key_wait");
        sem_krot.wait_min(1);
        sem_knorm.wait_min(1);
        invalidate_l1_cache();
    }

    for (uint32_t i = 0; i < lane_count; ++i) {
        FUSED_ZONE("fz_qs_mt_rs_lane");
        const uint32_t lane = lane_first + i;
        const uint32_t slot = positions[lane] & 31;
        cb_reserve_back(CB_STG, PACK_TILES);
        const uint32_t stg_l1 = get_write_ptr(CB_STG);
        for (uint32_t c = 0; c < PACK_TILES; ++c) {
            noc_async_read_page(lane * PACK_TILES + c, stg, stg_l1 + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
        for (uint32_t c = 0; c < PACK_TILES; ++c) {
            copy_tile_row(pack_l1 + c * TILE_BYTES, lane, stg_l1 + c * TILE_BYTES, slot, true);
        }
        cb_push_back(CB_STG, PACK_TILES);
    }
    sem_krot.set(0);
    sem_knorm.set(0);
}
