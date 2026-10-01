// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core writer: hands the normalized row's RoPE tiles (0, 1 and swapped) to its rope core.  A query
// head (ROLE 0) then writes, per lane, the head's sparse-query row: 512 zero bytes, the normalized tiles 2..7 at
// bytes 640..1023 (the rope core fills 512..639), and zero rows for the heads 6, 12, .. of its residue class.  The
// key (ROLE 1) hands its tiles 2..7 to every staging core instead (one staging core per lane).
// Compile-time args: 0 ROLE, then TensorAccessorArgs query.
// Runtime args: 0 query address, 1 rows, 2 head, 3 rope core x, 4 rope core y, 5 staging core count, then the staging
// cores' (x, y) pairs.

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
    const uint32_t rope_x = get_arg_val<uint32_t>(3);
    const uint32_t rope_y = get_arg_val<uint32_t>(4);
    const uint32_t staging_cores = get_arg_val<uint32_t>(5);
    constexpr uint32_t ROLE = get_compile_time_arg_val(0);
    constexpr auto query_args = TensorAccessorArgs<1>();
    const auto query = TensorAccessor(query_args, query_addr);
    Noc noc;
    Semaphore<> sem_rope(SEM_ROPE), sem_knorm(SEM_KNORM);

    cb_wait_front(CB_N, HEAD_TILES);
    const uint32_t l1 = get_read_ptr(CB_N);
    {
        FUSED_ZONE("fz_qs_mt_wn_handoff");
        noc_async_write(l1, get_noc_addr(rope_x, rope_y, get_write_ptr(CB_IN)), ROPE_TILES * TILE_BYTES);
        noc_async_write(l1 + TILE_BYTES, get_noc_addr(rope_x, rope_y, get_write_ptr(CB_ROT)), TILE_BYTES);
        noc_async_write(l1, get_noc_addr(rope_x, rope_y, get_write_ptr(CB_ROT) + TILE_BYTES), TILE_BYTES);
        noc_async_write_barrier();
        sem_rope.up(noc, rope_x, rope_y, 1);
    }

    if constexpr (ROLE == 0) {
        FUSED_ZONE("fz_qs_mt_wn_query");
        cb_wait_front(CB_ZERO, 1);
        const uint32_t zero = get_read_ptr(CB_ZERO);
        for (uint32_t lane = 0; lane < rows; ++lane) {
            const uint32_t page = head * rows + lane;
            noc_async_write(zero, query.get_noc_addr(page, 0), HEAD_ROW_BYTES);
            for (uint32_t c = ROPE_TILES; c < HEAD_TILES; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        l1 + c * TILE_BYTES + chunk_offset(lane, half),
                        query.get_noc_addr(page, HEAD_ROW_BYTES + c * 2 * ROW_BYTES + half * ROW_BYTES),
                        ROW_BYTES);
                }
            }
            for (uint32_t zero_head = head + LOCAL_HEADS; zero_head < QUERY_HEADS; zero_head += LOCAL_HEADS) {
                noc_async_write(zero, query.get_noc_addr(zero_head * rows + lane, 0), QUERY_ROW_BYTES);
            }
        }
        noc_async_write_barrier();
    } else {
        FUSED_ZONE("fz_qs_mt_wn_key");
        for (uint32_t s = 0; s < staging_cores; ++s) {
            const uint32_t st_x = get_arg_val<uint32_t>(6 + 2 * s), st_y = get_arg_val<uint32_t>(7 + 2 * s);
            noc_async_write(
                l1 + ROPE_TILES * TILE_BYTES,
                get_noc_addr(st_x, st_y, get_write_ptr(CB_PACK) + (HEAD_TILES + ROPE_TILES) * TILE_BYTES),
                (HEAD_TILES - ROPE_TILES) * TILE_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < staging_cores; ++s) {
            sem_knorm.up(noc, get_arg_val<uint32_t>(6 + 2 * s), get_arg_val<uint32_t>(7 + 2 * s), 1);
        }
    }
    cb_pop_front(CB_N, HEAD_TILES);
    noc_async_atomic_barrier();
}
