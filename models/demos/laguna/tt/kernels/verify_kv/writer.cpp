// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Verify KV write (Laguna; verify_kv_write.py), writer. Per head, compute untilizes the cache tile into
// cb_untilized_cache; the writer copies the new row(s) into it, hands it back as cb_untilized_cache2 (same L1) for
// compute to tilize into cb_out, and writes cb_out to the cache. Roles (see reader.cpp): a follower writes no cache
// tile; it tells its leader once its row is untilized (semaphore 0 on the leader += 1) and keeps the row in L1 until
// the leader has copied it (semaphore 1 on the follower). The leader waits for all its followers, copies its own row
// and each follower's row (the same untilized-input L1 address on the follower's core) into their rows of the tile,
// writes the tile once and releases the followers.
//
// Adapted from paged_cache's writer_paged_fused_update_cache_interleaved_start_id.cpp: the legacy head loop
// (78608b96a50^, lines 125-172) and the group-mode exchange of jerrywangTT/laguna-s-merge-main (8437ef46690,
// GROUP_MODE blocks).
//
// Compile-time args: cb_out, cb_untilized_cache, cb_untilized_cache2, cb_untilized_input, cb_meta, Wt, Wbytes,
// num_heads, St, num_rows, sem_ready, sem_done, then TensorAccessorArgs of the cache.
// Runtime args: set (0 = K, 1 = V), row. Common runtime args: K cache address, V cache address, then the NOC (x, y) of
// every K row's core (num_rows pairs, in row order), then of every V row's core.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_untilized_cache = get_compile_time_arg_val(1);
    constexpr uint32_t cb_untilized_cache2 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_untilized_input = get_compile_time_arg_val(3);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);
    constexpr uint32_t Wbytes = get_compile_time_arg_val(6);
    constexpr uint32_t num_heads = get_compile_time_arg_val(7);
    constexpr uint32_t St = get_compile_time_arg_val(8);
    constexpr uint32_t num_rows = get_compile_time_arg_val(9);
    constexpr uint32_t sem_ready_id = get_compile_time_arg_val(10);
    constexpr uint32_t sem_done_id = get_compile_time_arg_val(11);
    constexpr auto cache_args = TensorAccessorArgs<12>();

    constexpr uint32_t head_offset_t = Wt * St;
    constexpr uint32_t SKIP = 0xFFFFFFFFu;

    const uint32_t set = get_arg_val<uint32_t>(0);
    const auto cache = TensorAccessor(cache_args, get_common_arg_val<uint32_t>(set == 0 ? 0 : 1));
    // this set's row cores: common args 2 + 2 * (set * num_rows + row) and the next one
    const uint32_t xy_base = 2 + 2 * set * num_rows;

    cb_wait_front(cb_meta, 1);
    volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
    const bool is_leader = meta[0] == 1;
    const uint32_t followers = is_leader ? meta[1] : 0;
    const uint32_t leader_row = meta[2];
    uint32_t cache_id = meta[3];
    const uint32_t own_offset_B = meta[4] * Wbytes;

    const uint32_t sem_ready = get_semaphore(sem_ready_id);
    const uint32_t sem_done = get_semaphore(sem_done_id);

    cb_wait_front(cb_untilized_input, Wt);  // this row, untilized: one Wbytes row per head
    if (is_leader) {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready);
        noc_semaphore_wait(ready, followers);
        noc_semaphore_set(ready, 0);
    } else if (leader_row != SKIP) {
        noc_semaphore_inc(
            get_noc_addr(
                get_common_arg_val<uint32_t>(xy_base + 2 * leader_row),
                get_common_arg_val<uint32_t>(xy_base + 2 * leader_row + 1),
                sem_ready),
            1);
        noc_async_atomic_barrier();
    }

    uint32_t input_l1_read_addr = get_read_ptr(cb_untilized_input);
    for (uint32_t h = 0; h < num_heads; ++h) {
        // Wait on compute to untilize a block. Update that block in L1.
        cb_wait_front(cb_untilized_cache, Wt);
        cb_reserve_back(cb_untilized_cache2, Wt);

        const uint32_t tile_rows_addr = get_read_ptr(cb_untilized_cache);
        noc_async_read(get_noc_addr(input_l1_read_addr), tile_rows_addr + own_offset_B, Wbytes);
        // the followers' rows: the same untilized-input address on their cores (one program, one buffer layout)
        for (uint32_t f = 0; f < followers; ++f) {
            const uint32_t row = meta[5 + 2 * f];
            const uint32_t row_in_tile = meta[6 + 2 * f];
            noc_async_read(
                get_noc_addr(
                    get_common_arg_val<uint32_t>(xy_base + 2 * row),
                    get_common_arg_val<uint32_t>(xy_base + 2 * row + 1),
                    input_l1_read_addr),
                tile_rows_addr + row_in_tile * Wbytes,
                Wbytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_untilized_cache2, Wt);
        cb_pop_front(cb_untilized_cache, Wt);

        // Wait on compute to tilize the updated block. Write it to the cache (leader only).
        cb_wait_front(cb_out, Wt);
        if (is_leader) {
            uint32_t out_l1_read_addr = get_read_ptr(cb_out);
            for (uint32_t t = 0; t < Wt; ++t) {
                noc_async_write_page(cache_id + t, cache, out_l1_read_addr);
                out_l1_read_addr += cache.get_aligned_page_size();
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(cb_out, Wt);
        if (is_leader) {
            noc_async_write_barrier();
        }

        input_l1_read_addr += Wbytes;
        cache_id += head_offset_t;
    }

    if (!is_leader && leader_row != SKIP) {
        // keep this row in L1 until the leader has copied it
        volatile tt_l1_ptr uint32_t* done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done);
        noc_semaphore_wait(done, 1);
        noc_semaphore_set(done, 0);
    }
    cb_pop_front(cb_untilized_input, Wt);
    for (uint32_t f = 0; f < followers; ++f) {
        const uint32_t row = meta[5 + 2 * f];
        noc_semaphore_inc(
            get_noc_addr(
                get_common_arg_val<uint32_t>(xy_base + 2 * row),
                get_common_arg_val<uint32_t>(xy_base + 2 * row + 1),
                sem_done),
            1);
    }
    noc_async_atomic_barrier();
    cb_pop_front(cb_meta, 1);
}
