// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads `n_tiles` tiles from interleaved DRAM into CB_in.
//
// Runtime args:
//   0: src_addr
//   1: n_tiles                (logical tiles to read; must be a multiple of tiles_per_page)
//   2: start_tile_id          (in tiles)
//   3: program_id
//   4: push_tile_count        (chunk size in tiles for modes 0/1; ignored for mode 2)
//   5: reader_mode            (0 = reserve N, read+push one-by-one with a global barrier
//                              1 = reserve N, read all, single barrier, push N
//                              2 = per-trid double buffer: two slots, alternating trids 1 and 2,
//                                  barrier on one trid while the other read is in flight)
//   6: tiles_per_page         (DRAM page size in tiles; one noc_async_read_tile pulls a page
//                              = this many tiles in one NoC transaction; CB still has 1 tile
//                              per page, so we push this many CB pages per DRAM page read)
//
// Profiler on first tile of the slice only: READ_BEFORE_BARRIER / READ_AFTER_BARRIER.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);
    const uint32_t push_tile_count = get_arg_val<uint32_t>(4);
    const uint32_t reader_mode = get_arg_val<uint32_t>(5);
    const uint32_t tiles_per_page_rt = get_arg_val<uint32_t>(6);
    const uint32_t tiles_per_page = tiles_per_page_rt > 0 ? tiles_per_page_rt : 1;

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr auto src_args = TensorAccessorArgs<1>();

    const auto src = TensorAccessor(src_args, src_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("NCRISC_GO", program_id);

    const uint32_t chunk_size = push_tile_count > 0 ? push_tile_count : 1;
    const uint32_t end_tile_id = start_tile_id + n_tiles;
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;

    // DRAM page = tiles_per_page tiles (one NoC transaction per page).
    // src accessor is configured with the matching page_size at build time.
    const uint32_t start_page_id = start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;

    if (reader_mode == 2) {
        // Per-trid double-buffer: exactly two slots in the CB (depth=2),
        // alternating trids 1 and 2 (slot 0 always uses TRID_A, slot 1 always
        // uses TRID_B). We track the two slot addresses explicitly because
        // get_write_ptr cannot tell us where the in-flight read is going.
        //
        // Sequence:
        //   issue read tile 0 -> slot 0 (TRID_A)
        //   issue read tile 1 -> slot 1 (TRID_B)
        //   loop:
        //     barrier(TRID_A) -> push slot 0 -> compute consumes tile 0
        //     wait for slot 0 to be free (cb_reserve_back gates on consumer)
        //     issue next tile -> slot 0 (TRID_A) — overlaps with compute
        //     barrier(TRID_B) -> push slot 1 -> compute consumes tile 1
        //     wait for slot 1 to be free
        //     issue next tile -> slot 1 (TRID_B)
        constexpr uint32_t TRID_A = 1;
        constexpr uint32_t TRID_B = 2;
        const uint32_t trids[2] = {TRID_A, TRID_B};

        if (n_pages == 0) {
            DeviceTimestampedData("NCRISC_DONE", program_id);
            return;
        }

        const uint32_t page_bytes = tiles_per_page * tile_bytes;
        // Each slot holds one DRAM page (= tiles_per_page CB pages).
        const uint32_t initial_fill_pages = n_pages >= 2 ? 2 : 1;
        cb_reserve_back(cb_in, initial_fill_pages * tiles_per_page);
        const uint32_t cb_base = get_write_ptr(cb_in);
        const uint32_t slot_addrs[2] = {cb_base, cb_base + page_bytes};

        DeviceTimestampedData("READ_BEFORE_BARRIER", start_tile_id);

        noc_async_read_set_trid(TRID_A);
        noc_async_read_tile(start_page_id, src, slot_addrs[0]);
        if (initial_fill_pages == 2) {
            noc_async_read_set_trid(TRID_B);
            noc_async_read_tile(start_page_id + 1, src, slot_addrs[1]);
        }

        uint32_t next_page_id = start_page_id + initial_fill_pages;
        uint32_t pages_pushed = 0;
        uint32_t slot = 0;

        while (pages_pushed < n_pages) {
            const uint32_t trid = trids[slot];
            noc_async_read_barrier_with_trid(trid);
            if (pages_pushed == 0) {
                DeviceTimestampedData("READ_AFTER_BARRIER", start_tile_id);
            }
            cb_push_back(cb_in, tiles_per_page);
            ++pages_pushed;

            if (next_page_id < end_page_id) {
                cb_reserve_back(cb_in, tiles_per_page);
                noc_async_read_set_trid(trid);
                noc_async_read_tile(next_page_id, src, slot_addrs[slot]);
                ++next_page_id;
            }
            slot ^= 1;
        }

        DeviceTimestampedData("NCRISC_DONE", program_id);
        return;
    }

    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id;) {
        const uint32_t tiles_left = end_tile_id - tile_id;
        const uint32_t chunk = tiles_left < chunk_size ? tiles_left : chunk_size;

        cb_reserve_back(cb_in, chunk);
        const uint32_t cb_base = get_write_ptr(cb_in);

        if (reader_mode == 1) {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = cb_base + i * tile_bytes;
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
            }
            cb_push_back(cb_in, chunk);
        } else {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = get_write_ptr(cb_in);
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
                cb_push_back(cb_in, 1);
            }
        }

        tile_id += chunk;
    }

    DeviceTimestampedData("NCRISC_DONE", program_id);
}
