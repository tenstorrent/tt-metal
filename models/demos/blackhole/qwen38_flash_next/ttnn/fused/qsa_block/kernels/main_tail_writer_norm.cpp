// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, norm core writer: hands the normalized row's RoPE tiles (0, 1 and swapped) to its rope core.  A query
// head (role 0) then writes, per lane, the head's sparse-query row: 512 zero bytes, the normalized tiles 2..7 at
// bytes 640..1023 (the rope core fills 512..639), and zero rows for the heads 6, 12, .. of its residue class.  The
// key (role 1) hands its tiles 2..7 to every staging core instead.
// Compile-time args: 0 role, then TensorAccessorArgs query.
// Runtime args: 0 query address, 1 rows, 2 head, 3 rope core x, 4 rope core y, 5 staging core count, then the staging
// cores' (x, y) pairs.

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
    const uint32_t rope_x = get_arg_val<uint32_t>(3);
    const uint32_t rope_y = get_arg_val<uint32_t>(4);
    const uint32_t staging_cores = get_arg_val<uint32_t>(5);
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr auto query_args = TensorAccessorArgs<1>();
    const auto query = TensorAccessor(query_args, query_addr);
    Noc noc;
    Semaphore<> rope_handoff(rope_semaphore_id), key_normalized_handoff(key_normalized_semaphore_id);

    CircularBuffer(cb_n).wait_front(head_tiles);
    const uint32_t normed_l1 = get_read_ptr(cb_n);
    {
        noc_async_write(normed_l1, get_noc_addr(rope_x, rope_y, get_write_ptr(cb_in)), rope_tiles * tile_bytes);
        noc_async_write(normed_l1 + tile_bytes, get_noc_addr(rope_x, rope_y, get_write_ptr(cb_rot)), tile_bytes);
        noc_async_write(normed_l1, get_noc_addr(rope_x, rope_y, get_write_ptr(cb_rot) + tile_bytes), tile_bytes);
        noc_async_write_barrier();
        rope_handoff.up(noc, rope_x, rope_y, 1);
    }

    if constexpr (role == 0) {
        CircularBuffer(cb_zero).wait_front(1);
        const uint32_t zero_l1 = get_read_ptr(cb_zero);
        for (uint32_t lane = 0; lane < rows; ++lane) {
            const uint32_t page = head * rows + lane;
            noc_async_write(zero_l1, query.get_noc_addr(page, 0), head_row_bytes);
            for (uint32_t c = rope_tiles; c < head_tiles; ++c) {
                for (uint32_t half = 0; half < 2; ++half) {
                    noc_async_write(
                        normed_l1 + c * tile_bytes + chunk_offset(lane, half),
                        query.get_noc_addr(page, head_row_bytes + c * 2 * half_row_bytes + half * half_row_bytes),
                        half_row_bytes);
                }
            }
            for (uint32_t zero_head = head + local_heads; zero_head < query_heads; zero_head += local_heads) {
                noc_async_write(zero_l1, query.get_noc_addr(zero_head * rows + lane, 0), query_row_bytes);
            }
        }
        noc_async_write_barrier();
    } else {
        for (uint32_t s = 0; s < staging_cores; ++s) {
            const uint32_t staging_x = get_arg_val<uint32_t>(6 + 2 * s), staging_y = get_arg_val<uint32_t>(7 + 2 * s);
            noc_async_write(
                normed_l1 + rope_tiles * tile_bytes,
                get_noc_addr(staging_x, staging_y, get_write_ptr(cb_pack) + (head_tiles + rope_tiles) * tile_bytes),
                (head_tiles - rope_tiles) * tile_bytes);
        }
        noc_async_write_barrier();
        for (uint32_t s = 0; s < staging_cores; ++s) {
            key_normalized_handoff.up(noc, get_arg_val<uint32_t>(6 + 2 * s), get_arg_val<uint32_t>(7 + 2 * s), 1);
        }
    }
    CircularBuffer(cb_n).pop_front(head_tiles);
    noc_async_atomic_barrier();
}
