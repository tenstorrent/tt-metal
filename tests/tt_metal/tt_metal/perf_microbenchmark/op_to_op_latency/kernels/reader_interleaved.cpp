// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads `n_tiles` tiles from interleaved DRAM into CB_in.
//
// Runtime args:
//   0: src_addr
//   1: n_tiles                (logical tiles to read; must be a multiple of TILES_PER_PAGE)
//   2: start_tile_id          (in tiles)
//   3: program_id
//
// Compile-time args (see TensorAccessorArgs<1> for src accessor):
//   0: cb_in
//   1: READER_MODE            (0 = reserve N, read+push one-by-one with a global barrier
//                              1 = reserve N, read all, single barrier, push N
//                              2 = per-trid double-buffer: keep TRID_IN_FLIGHT reads in flight
//                                  under TRID_A and TRID_B, barrier on a TRID drains all its
//                                  reads, then refill before switching to the other TRID)
//   2: PUSH_TILE_COUNT        (chunk size in tiles for modes 0/1; ignored for mode 2)
//   3: TILES_PER_PAGE         (DRAM page size in tiles; one noc_async_read_tile pulls a page
//                              = this many tiles in one NoC transaction; CB still has 1 tile
//                              per page, so we push this many CB pages per DRAM page read)
//   4: TRID_IN_FLIGHT         (reads in flight per TRID for mode 2; CB depth must be
//                              >= 2 * TRID_IN_FLIGHT * TILES_PER_PAGE)
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

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t READER_MODE = get_compile_time_arg_val(1);
    constexpr uint32_t PUSH_TILE_COUNT = get_compile_time_arg_val(2);
    constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(3);
    constexpr uint32_t TRID_IN_FLIGHT = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();

    constexpr uint32_t chunk_size = PUSH_TILE_COUNT > 0 ? PUSH_TILE_COUNT : 1;
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr uint32_t trid_in_flight = TRID_IN_FLIGHT > 0 ? TRID_IN_FLIGHT : 1;

    const auto src = TensorAccessor(src_args, src_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("NCRISC_GO", program_id);

    const uint32_t end_tile_id = start_tile_id + n_tiles;
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;

    // DRAM page = tiles_per_page tiles (one NoC transaction per page).
    // src accessor is configured with the matching page_size at build time.
    const uint32_t start_page_id = start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;

    if constexpr (READER_MODE == 2) {
        // Per-trid double-buffer with N = TRID_IN_FLIGHT reads in flight per TRID.
        //
        // Slot layout (depth = 2*N*tiles_per_page, asserted on host):
        //   slots [0..N-1]     are owned by TRID_A
        //   slots [N..2N-1]    are owned by TRID_B
        //
        // Why this works with the CB:
        //   cb_reserve_back / cb_push_back only track a FIFO count (depth - filled);
        //   they don't know which physical slot we're writing. But since we always
        //   push in strict A, B, A, B, ... order, the consumer pops in that same
        //   order. So when cb_reserve_back(N*tpp) unblocks we know the N "oldest"
        //   slots have been freed - and those are exactly the slots of whichever
        //   side we are about to refill.
        //
        // Sequence:
        //   1. cb_reserve_back(2*N*tpp), issue N TRID_A reads to slots 0..N-1,
        //      issue N TRID_B reads to slots N..2N-1.
        //   2. barrier(A), push N tiles_per_page pages.   <-- consumer starts eating A
        //   3. barrier(B), push N tiles_per_page pages.   <-- consumer keeps eating
        //   4. Steady loop:
        //        cb_reserve_back(N*tpp)  // wait for consumer to free A's slots
        //        refill N TRID_A reads to slots 0..N-1
        //        barrier(A), push N
        //        cb_reserve_back(N*tpp)  // wait for consumer to free B's slots
        //        refill N TRID_B reads to slots N..2N-1
        //        barrier(B), push N
        constexpr uint32_t TRID_A = 1;
        constexpr uint32_t TRID_B = 2;
        constexpr uint32_t N = trid_in_flight;

        const uint32_t page_bytes = tiles_per_page * tile_bytes;

        // Initial reservation: up to 2N pages worth (clipped by what we'll actually fill).
        const uint32_t initial_fill_pages = (n_pages < 2u * N) ? n_pages : (2u * N);
        cb_reserve_back(cb_in, initial_fill_pages * tiles_per_page);
        const uint32_t cb_base = get_write_ptr(cb_in);

        DeviceTimestampedData("READ_BEFORE_BARRIER", start_tile_id);

        uint32_t next_page_id = start_page_id;
        uint32_t pages_pushed = 0;

        // Initial fill: up to N TRID_A reads to slots 0..N-1.
        uint32_t a_count = 0;
        noc_async_read_set_trid(TRID_A);
        while (a_count < N && next_page_id < end_page_id) {
            noc_async_read_tile(next_page_id, src, cb_base + a_count * page_bytes);
            ++next_page_id;
            ++a_count;
        }
        // Initial fill: up to N TRID_B reads to slots N..2N-1.
        uint32_t b_count = 0;
        noc_async_read_set_trid(TRID_B);
        while (b_count < N && next_page_id < end_page_id) {
            noc_async_read_tile(next_page_id, src, cb_base + (N + b_count) * page_bytes);
            ++next_page_id;
            ++b_count;
        }

        // Push initial A, then initial B (consumer can start at full bandwidth).
        if (a_count > 0) {
            noc_async_read_barrier_with_trid(TRID_A);
            DeviceTimestampedData("READ_AFTER_BARRIER", start_tile_id);
            cb_push_back(cb_in, a_count * tiles_per_page);
            pages_pushed += a_count;
        } else {
            DeviceTimestampedData("READ_AFTER_BARRIER", start_tile_id);
        }
        if (b_count > 0) {
            noc_async_read_barrier_with_trid(TRID_B);
            cb_push_back(cb_in, b_count * tiles_per_page);
            pages_pushed += b_count;
        }

        // Steady-state: alternately refill A, push A, refill B, push B.
        // Each refill issues up to N more reads on its TRID, overlapping with the
        // other TRID's in-flight reads and the consumer's pop on the previous batch.
        while (pages_pushed < n_pages) {
            // Refill A. cb_reserve_back blocks until consumer has popped at least N
            // tiles past the previous A push, freeing slots 0..N-1.
            cb_reserve_back(cb_in, N * tiles_per_page);
            uint32_t a_refill = 0;
            noc_async_read_set_trid(TRID_A);
            while (a_refill < N && next_page_id < end_page_id) {
                noc_async_read_tile(next_page_id, src, cb_base + a_refill * page_bytes);
                ++next_page_id;
                ++a_refill;
            }
            if (a_refill > 0) {
                noc_async_read_barrier_with_trid(TRID_A);
                cb_push_back(cb_in, a_refill * tiles_per_page);
                pages_pushed += a_refill;
            }
            if (pages_pushed >= n_pages) {
                break;
            }

            // Refill B. Same idea on slots N..2N-1.
            cb_reserve_back(cb_in, N * tiles_per_page);
            uint32_t b_refill = 0;
            noc_async_read_set_trid(TRID_B);
            while (b_refill < N && next_page_id < end_page_id) {
                noc_async_read_tile(next_page_id, src, cb_base + (N + b_refill) * page_bytes);
                ++next_page_id;
                ++b_refill;
            }
            if (b_refill > 0) {
                noc_async_read_barrier_with_trid(TRID_B);
                cb_push_back(cb_in, b_refill * tiles_per_page);
                pages_pushed += b_refill;
            }
        }

        DeviceTimestampedData("NCRISC_DONE", program_id);
        return;
    }

    // Legacy reader modes (0 = incremental push-1; 1 = batch read+push).
    for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id;) {
        const uint32_t tiles_left = end_tile_id - tile_id;
        const uint32_t chunk = tiles_left < chunk_size ? tiles_left : chunk_size;

        cb_reserve_back(cb_in, chunk);
        const uint32_t cb_base = get_write_ptr(cb_in);

        if constexpr (READER_MODE == 1) {
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
