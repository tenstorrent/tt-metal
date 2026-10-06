// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core writer: a query head (ROLE 0) writes, per lane, the rotated tiles' row into bytes 512..639 of
// the head's sparse-query row; the key (ROLE 1) hands its rotated tiles to every staging core (pack tiles 8, 9; one
// staging core per lane).
// Compile-time args: 0 ROLE, then TensorAccessorArgs query.
// Runtime args: 0 query address, 1 rows, 2 head, 3 staging core count, then the staging cores' (x, y) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"
#include "../../kernels/zones.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t query_addr = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t staging_cores = get_arg_val<uint32_t>(3);
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr auto query_args = TensorAccessorArgs<1>();
    const auto query = TensorAccessor(query_args, query_addr);
    Noc noc;
    Semaphore<> sem_krot(SEM_KROT);

    cb_wait_front(CB_OUT, ROPE_TILES);
    const uint32_t l1 = get_read_ptr(CB_OUT);
    if constexpr (ROLE == 0) {
        FUSED_ZONE("fz_qs_mt_wr_query");
        for (uint32_t lane = 0; lane < rows; ++lane) {
            const uint32_t page = head * rows + lane;
            for (uint32_t t = 0; t < ROPE_TILES; ++t) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        l1 + t * TILE_BYTES + chunk_offset(lane, half),
                        query.get_noc_addr(page, HEAD_ROW_BYTES + t * 2 * ROW_BYTES + half * ROW_BYTES),
                        ROW_BYTES);
                }
            }
        }
        noc_async_write_barrier();
    } else {
        FUSED_ZONE("fz_qs_mt_wr_key");
        for (uint32_t s = 0; s < staging_cores; ++s) {
            const uint32_t st_x = get_arg_val<uint32_t>(4 + 2 * s), st_y = get_arg_val<uint32_t>(5 + 2 * s);
            noc_async_write(
                l1,
                get_noc_addr(st_x, st_y, get_write_ptr(CB_PACK) + HEAD_TILES * TILE_BYTES),
                ROPE_TILES * TILE_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < staging_cores; ++s) {
            sem_krot.up(noc, get_arg_val<uint32_t>(4 + 2 * s), get_arg_val<uint32_t>(5 + 2 * s), 1);
        }
    }
    cb_pop_front(CB_OUT, ROPE_TILES);
    noc_async_atomic_barrier();
}
