// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, one-row staging core writer (QWEN38_QSA_MT_ROW=1, one lane, unpaged): the new row (CB_ROW: two 512 B
// pages, the V half then the K half, canonicalized by the reader) to the KV cache row lane * lane_rows + P (V half at
// byte 0, K half at byte 512 of the row [V | K]), and into row (P % 32) of each of the 16 staging tiles (two 32-byte
// chunks per tile); each half is written when the reader pushes it.  The other staging rows and cache rows are not
// touched.  One write barrier.
// Compile-time args: TensorAccessorArgs staging, cache, kv_block_start, kv_row_hit.  Runtime args: 0 staging, 1 cache,
// 2 kv_block_start, 3 kv_row_hit addresses, 4 rows, 5 lane rows (cache rows per lane), 6 lane.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "main_tail_row.h"
#include "positions.h"
#include "api/dataflow/noc_semaphore.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t stg_addr = get_arg_val<uint32_t>(0);
    const uint32_t cache_addr = get_arg_val<uint32_t>(1);
    const uint32_t pos_addr = get_arg_val<uint32_t>(2);
    const uint32_t hit_addr = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    const uint32_t lane_rows = get_arg_val<uint32_t>(5);
    const uint32_t lane = get_arg_val<uint32_t>(6);
    constexpr auto stg_args = TensorAccessorArgs<0>();
    constexpr auto cache_args = TensorAccessorArgs<stg_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cache_args.next_compile_time_args_offset()>();
    constexpr auto hit_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    const auto stg = TensorAccessor(stg_args, stg_addr);
    const auto cache = TensorAccessor(cache_args, cache_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);
    const auto hit = TensorAccessor(hit_args, hit_addr);

    cb_reserve_back(CB_POS_W, 1);
    cb_reserve_back(CB_POSCW, 1);
    const uint32_t pos_l1 = get_write_ptr(CB_POS_W);
    {
        FUSED_ZONE("fz_qs_mt_ws_pos");
        tile_rows::read_positions(
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1), pos, hit, rows, get_write_ptr(CB_POSCW));
    }
    const uint32_t p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_l1)[lane];

    const uint32_t slot = p & 31;
    const uint32_t cache_row = lane * lane_rows + p;
    constexpr uint32_t HALF_BYTES = QUERY_ROW_BYTES / 2;
    uint32_t row = 0;
    // page 0 = the V half (row bytes 0..511), page 1 = the K half (512..1023): each half to its cache bytes and its 16
    // staging chunks as soon as the reader pushes it
    for (uint32_t h = 0; h < 2; ++h) {
        if (h == 1) {  // build tiles 12..15 of the K half here (the reader does 8..11), after the key tiles arrived
            FUSED_ZONE("fz_qs_mt_ws_khalf");
            Semaphore<> sem_krot(SEM_KROT), sem_knorm(SEM_KNORM), sem_done(SEM_ROW_DONE);
            sem_krot.wait_min(1);
            sem_knorm.wait_min(1);
            invalidate_l1_cache();
            build_tiles(
                get_write_ptr(CB_PACK),
                HEAD_TILES + HEAD_TILES / 2,
                HEAD_TILES / 2,
                lane,
                row + HALF_BYTES + HALF_BYTES / 2);
            sem_done.set(1);
        }
        {
            FUSED_ZONE("fz_qs_mt_ws_row_wait");
            cb_wait_front(CB_ROW, h + 1);
        }
        if (h == 0) {
            row = get_read_ptr(CB_ROW);
        }
        FUSED_ZONE("fz_qs_mt_ws_lane");
        const uint32_t half_l1 = row + h * HALF_BYTES;
        {
            FUSED_ZONE("fz_qs_mt_ws_cache_write");
            noc_async_write(half_l1, cache.get_noc_addr(cache_row, h * HALF_BYTES), HALF_BYTES);
        }
        {
            FUSED_ZONE("fz_qs_mt_ws_stg_issue");
            for (uint32_t t = 0; t < HEAD_TILES; ++t) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        half_l1 + t * 2 * ROW_BYTES + half * ROW_BYTES,
                        stg.get_noc_addr(lane * PACK_TILES + h * HEAD_TILES + t, chunk_offset(slot, half)),
                        ROW_BYTES);
                }
            }
        }
    }
    {
        FUSED_ZONE("fz_qs_mt_ws_barrier");
        noc_async_write_barrier();
    }
    cb_pop_front(CB_ROW, 2);
}
