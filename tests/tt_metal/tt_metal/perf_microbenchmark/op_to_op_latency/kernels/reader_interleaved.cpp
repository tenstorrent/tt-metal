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
// Compile-time args (see TensorAccessorArgs<6> for src accessor):
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
//   5: CROSS_PROGRAM_OFFSET_TILES (0 = every program reads the same slice; >0 = program k
//                              reads pages starting at start_tile_id + k*OFFSET. Host
//                              must allocate a buffer big enough for all program slices.)
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
    constexpr uint32_t CROSS_PROGRAM_OFFSET_TILES = get_compile_time_arg_val(5);
    constexpr auto src_args = TensorAccessorArgs<6>();

    constexpr uint32_t chunk_size = PUSH_TILE_COUNT > 0 ? PUSH_TILE_COUNT : 1;
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr uint32_t trid_in_flight = TRID_IN_FLIGHT > 0 ? TRID_IN_FLIGHT : 1;

    const auto src = TensorAccessor(src_args, src_addr);

    DeviceTimestampedData("PROG_ID", program_id);
    DeviceTimestampedData("NCRISC_GO", program_id);

    // When CROSS_PROGRAM_OFFSET_TILES > 0, each program reads a disjoint tile slice
    // so the DRAM controller sees fresh row opens per program (more app-like).
    const uint32_t effective_start_tile_id = start_tile_id + program_id * CROSS_PROGRAM_OFFSET_TILES;
    const uint32_t end_tile_id = effective_start_tile_id + n_tiles;
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;

    // DRAM page = tiles_per_page tiles (one NoC transaction per page).
    // src accessor is configured with the matching page_size at build time.
    const uint32_t start_page_id = effective_start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;

    if constexpr (READER_MODE == 2) {
        // Per-trid double buffer, N = TRID_IN_FLIGHT reads in flight per side.
        //
        // We are only measuring NoC read bandwidth: results are dummy and the compute
        // side is a NOP. We STRIDE the source page_id across [start_page_id, +n_pages)
        // (one increment per read) so consecutive transactions land on different DRAM
        // banks -- the interleaved buffer maps page k to bank k % num_banks. Reading a
        // single page pins every transaction to one bank and caps a single core at the
        // single-bank ceiling (~22 GB/s on WH); striding across all banks reaches the
        // ~30 GB/s all-bank ceiling (matches test_bw_and_latency -m 3 vs -m 1). Total
        // reads already equal n_pages, so the stride sweeps the core's slice exactly
        // once with no wrap. The L1 destination stays fixed at cb_base (dummy data; the
        // single-bank reference shows distinct L1 is not what gates BW).
        //
        // cb_reserve_back / cb_push_back are kept purely for flow control so the
        // consumer pipeline advances; host guarantees depth >= 2*N*tiles_per_page.
        //
        // Pipeline (one batch always in flight while we stall on the other):
        //   prologue: issue batch 0 on TRID_A.
        //   head    : prefetch batch 1 on TRID_B, then barrier+push batch 0 (one-shot,
        //             so the first-read profiler marker lives outside the hot loop).
        //   hot loop: prefetch the NEXT batch on the other trid, then barrier+push the
        //             oldest in-flight batch. The loop condition guarantees a next
        //             batch exists, so there is no per-iteration have_next branch, and
        //             the drained batch is always a full N (only the last batch can be
        //             partial). Static branch prediction on this arch makes the
        //             data-dependent branch worth eliminating.
        //   tail    : drain the final lone in-flight batch (may be < N).
        //
        // TRID_A ^ 1 == TRID_B and vice versa, so a single xor toggles sides.
        constexpr uint32_t TRID_A = 2;
        constexpr uint32_t TRID_B = 3;
        constexpr uint32_t N = trid_in_flight;
        const uint32_t batch_tiles = N * tiles_per_page;

        // Reserve room for both in-flight batches up front (covers prologue + head).
        cb_reserve_back(cb_in, 2 * batch_tiles);
        const uint32_t cb_base = get_write_ptr(cb_in);

        DeviceTimestampedData("READ_BEFORE_BARRIER", effective_start_tile_id);

        // Prologue: issue batch 0 on TRID_A.
        uint32_t issue_trid = TRID_A;
        uint32_t drain_trid = TRID_A;
        uint32_t page_id = start_page_id;  // strides across banks, one bump per read
        uint32_t issued = n_pages < N ? n_pages : N;
        noc_async_read_set_trid(issue_trid);
        for (uint32_t i = 0; i < issued; ++i) {
            noc_async_read_tile(page_id++, src, cb_base);
        }
        uint32_t pushed = 0;

        // Head: if a second batch exists, prefetch it on the other trid (so two batches
        // are in flight), then drain batch 0. The first-read marker is emitted here,
        // once, so it never enters the hot loop.
        if (issued < n_pages) {
            issue_trid ^= 1u;
            const uint32_t remaining = n_pages - issued;
            const uint32_t nb = remaining < N ? remaining : N;
            noc_async_read_set_trid(issue_trid);
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read_tile(page_id++, src, cb_base);
            }
            issued += nb;

            noc_async_read_barrier_with_trid(drain_trid);
            DeviceTimestampedData("READ_AFTER_BARRIER", effective_start_tile_id);
            cb_push_back(cb_in, batch_tiles);
            pushed += N;
            drain_trid ^= 1u;
        }

        // Hot loop: uniform prefetch-then-drain, no data-dependent branch. The reserve
        // gates reuse of the region the oldest batch just vacated.
        while (issued < n_pages) {
            issue_trid ^= 1u;
            const uint32_t remaining = n_pages - issued;
            const uint32_t nb = remaining < N ? remaining : N;
            cb_reserve_back(cb_in, batch_tiles);
            noc_async_read_set_trid(issue_trid);
            for (uint32_t i = 0; i < nb; ++i) {
                noc_async_read_tile(page_id++, src, cb_base);
            }
            issued += nb;

            noc_async_read_barrier_with_trid(drain_trid);
            cb_push_back(cb_in, batch_tiles);
            pushed += N;
            drain_trid ^= 1u;
        }

        // Tail: drain the final in-flight batch (may be partial).
        noc_async_read_barrier_with_trid(drain_trid);
        if (pushed == 0) {
            // n_pages <= N: head never ran, so emit the first-read marker here.
            DeviceTimestampedData("READ_AFTER_BARRIER", effective_start_tile_id);
        }
        cb_push_back(cb_in, (n_pages - pushed) * tiles_per_page);

        DeviceTimestampedData("NCRISC_DONE", program_id);
        return;
    }

    // Legacy reader modes (0 = incremental push-1; 1 = batch read+push).
    for (uint32_t tile_id = effective_start_tile_id; tile_id < end_tile_id;) {
        const uint32_t tiles_left = end_tile_id - tile_id;
        const uint32_t chunk = tiles_left < chunk_size ? tiles_left : chunk_size;

        cb_reserve_back(cb_in, chunk);
        const uint32_t cb_base = get_write_ptr(cb_in);

        if constexpr (READER_MODE == 1) {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = cb_base + i * tile_bytes;
                if (tid == effective_start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == effective_start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
            }
            cb_push_back(cb_in, chunk);
        } else {
            for (uint32_t i = 0; i < chunk; ++i) {
                const uint32_t tid = tile_id + i;
                const uint32_t l1_write_addr = get_write_ptr(cb_in);
                if (tid == effective_start_tile_id) {
                    DeviceTimestampedData("READ_BEFORE_BARRIER", tid);
                }
                noc_async_read_tile(tid, src, l1_write_addr);
                noc_async_read_barrier();
                if (tid == effective_start_tile_id) {
                    DeviceTimestampedData("READ_AFTER_BARRIER", tid);
                }
                cb_push_back(cb_in, 1);
            }
        }

        tile_id += chunk;
    }

    DeviceTimestampedData("NCRISC_DONE", program_id);
}
