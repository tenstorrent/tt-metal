// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, one-row staging core reader (QWEN38_QSA_MT_ROW=1, one lane): the value tiles into pack tiles 0..7 (the
// key's rotated tiles 8, 9 and normalized tiles 10..15 arrive from the key cores), then the new row [v | k] as one
// ROW_MAJOR 1 KB row in CB_ROW with the values canonicalized as the SFPU one-hot select of the default path stores them
// (see ``canonical``).  Neither the staging tiles nor the positions are read here: the writer reads the position and
// places the row. Compile-time args: TensorAccessorArgs v.  Runtime args: 0 v, 1 first tile of the value in v, 2 lane.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "main_tail_row.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t v_addr = get_arg_val<uint32_t>(0);
    const uint32_t v_first = get_arg_val<uint32_t>(1);
    const uint32_t lane = get_arg_val<uint32_t>(2);
    constexpr auto v_args = TensorAccessorArgs<0>();
    const auto v = TensorAccessor(v_args, v_addr);
    Semaphore<> sem_krot(SEM_KROT), sem_knorm(SEM_KNORM);
    Semaphore<> sem_done(SEM_ROW_DONE);
    constexpr uint32_t K_TILES = HEAD_TILES / 2;  // tiles 8..11 here, 12..15 on the writer

    cb_reserve_back(CB_PACK, PACK_TILES);
    cb_reserve_back(CB_ROW, 2);
    const uint32_t pack_l1 = get_write_ptr(CB_PACK), row_l1 = get_write_ptr(CB_ROW);
    {
        FUSED_ZONE("fz_qs_mt_rs_value");
        for (uint32_t c = 0; c < HEAD_TILES; ++c) {
            noc_async_read_page(v_first + c, v, pack_l1 + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    {
        FUSED_ZONE("fz_qs_mt_rs_vhalf");
        build_tiles(pack_l1, 0, HEAD_TILES, lane, row_l1);
        cb_push_back(CB_ROW, 1);
    }
    {
        FUSED_ZONE("fz_qs_mt_rs_key_wait");
        sem_krot.wait_min(1);
        sem_knorm.wait_min(1);
        invalidate_l1_cache();
    }
    {
        FUSED_ZONE("fz_qs_mt_rs_lane");
        build_tiles(pack_l1, HEAD_TILES, K_TILES, lane, row_l1 + QUERY_ROW_BYTES / 2);
        cb_push_back(CB_ROW, 1);
    }
    sem_done.wait(1);  // the writer has passed its own key wait and built its tiles: the semaphores can be reset
    sem_done.set(0);
    sem_krot.set(0);
    sem_knorm.set(0);
}
