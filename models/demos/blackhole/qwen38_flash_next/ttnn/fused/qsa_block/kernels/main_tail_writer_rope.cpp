// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, rope core writer: a query head (role 0) writes, per lane, the rotated tiles' row into bytes 512..639 of
// the head's sparse-query row; the key (role 1) hands its rotated tiles to every staging core (pack tiles 8, 9).
// Compile-time args: 0 role, then TensorAccessorArgs query.
// Runtime args: 0 query address, 1 rows, 2 head, 3 staging core count, then the staging cores' (x, y) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t query_addr = get_arg_val<uint32_t>(0);
    const uint32_t rows = get_arg_val<uint32_t>(1);
    const uint32_t head = get_arg_val<uint32_t>(2);
    const uint32_t staging_cores = get_arg_val<uint32_t>(3);
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr auto query_args = TensorAccessorArgs<1>();
    const auto query = TensorAccessor(query_args, query_addr);
    Noc noc;
    Semaphore<> key_rotated_handoff(key_rotated_semaphore_id);

    CircularBuffer(cb_out).wait_front(rope_tiles);
    const uint32_t rotated_l1 = get_read_ptr(cb_out);
    if constexpr (role == 0) {
        for (uint32_t lane = 0; lane < rows; ++lane) {
            const uint32_t page = head * rows + lane;
            for (uint32_t t = 0; t < rope_tiles; ++t) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        rotated_l1 + t * tile_bytes + chunk_offset(lane, half),
                        query.get_noc_addr(page, head_row_bytes + t * 2 * half_row_bytes + half * half_row_bytes),
                        half_row_bytes);
                }
            }
        }
        noc_async_write_barrier();
    } else {
        for (uint32_t s = 0; s < staging_cores; ++s) {
            const uint32_t staging_x = get_arg_val<uint32_t>(4 + 2 * s), staging_y = get_arg_val<uint32_t>(5 + 2 * s);
            noc_async_write(
                rotated_l1,
                get_noc_addr(staging_x, staging_y, get_write_ptr(cb_pack) + head_tiles * tile_bytes),
                rope_tiles * tile_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < staging_cores; ++s) {
            key_rotated_handoff.up(noc, get_arg_val<uint32_t>(4 + 2 * s), get_arg_val<uint32_t>(5 + 2 * s), 1);
        }
    }
    CircularBuffer(cb_out).pop_front(rope_tiles);
    noc_async_atomic_barrier();
}
