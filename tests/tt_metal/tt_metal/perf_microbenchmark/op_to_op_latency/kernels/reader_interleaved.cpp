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
//   4: read_start_delay       (--reader-stagger-cycles * core index; spin this many nops after
//                              go, before reading, to induce a controlled read-completion skew)
//   5: workload_repeat        (--kernel-unroll; repeat the whole read this many times inside ONE
//                              invocation, no barrier between reps; 0/1 = normal single pass)
//   6: read_progress_every    (mode 2: emit a READ_PROG timestamp every N pages completed; 0=off)
//
// Compile-time args (see TensorAccessorArgs<8> for src accessor):
//   0: cb_in
//   1: READER_MODE            (0 = reserve N, read+push one-by-one with a global barrier
//                              1 = reserve N, read all, single barrier, push N
//                              2 = per-trid double-buffer: keep TRID_IN_FLIGHT reads in flight
//                                  under TRID_A and TRID_B, barrier on a TRID drains all its
//                                  reads, then refill before switching to the other TRID)
//   2: PUSH_TILE_COUNT        (chunk size in tiles for modes 0/1; ignored for mode 2)
//   3: TILES_PER_PAGE         (DRAM page size in tiles; one async_read pulls a page
//                              = this many tiles in one NoC transaction; CB still has 1 tile
//                              per page, so we push this many CB pages per DRAM page read)
//   4: TRID_IN_FLIGHT         (reads in flight per TRID for mode 2; CB depth must be
//                              >= 2 * TRID_IN_FLIGHT * TILES_PER_PAGE)
//   5: CROSS_PROGRAM_OFFSET_TILES (0 = every program reads the same slice; >0 = program k
//                              reads pages starting at start_tile_id + k*OFFSET. Host
//                              must allocate a buffer big enough for all program slices.)
//   6: PROFILE_DETAIL          (0 = lean CI path: emit no custom markers; 1 = research:
//                              emit GO/DONE/BARRIER markers for BW/gap decomposition)
//   7: READ_BYTES_OVERRIDE     (reader_mode 2 only; 0 = read full page. >0 = NoC-read only this
//                              many bytes per page but still push a full CB page -- "cheap read"
//                              for the output-bound regime, valid only because payload is dummy.)
//
// Profiler on first tile of the slice only: READ_BEFORE_BARRIER / READ_AFTER_BARRIER
// (research only; gated by PROFILE_DETAIL).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "tools/profiler/kernel_profiler.hpp"

// Research-only detail markers. Compiled out on the lean CI path (PROFILE_DETAIL == 0) so
// they add zero device cycles to the gated op2op measurement. Expanded only inside
// kernel_main, where PROFILE_DETAIL (a compile-time arg) is in scope.
#define DETAIL_MARK(name, value)                \
    do {                                        \
        if constexpr (PROFILE_DETAIL) {         \
            DeviceTimestampedData(name, value); \
        }                                       \
    } while (0)

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);
    // Per-core read-start stagger: spin this many nops AFTER the (synchronized) go but BEFORE
    // issuing reads (host passes core_index * --reader-stagger-cycles). 0 = off.
    const uint32_t read_start_delay = get_arg_val<uint32_t>(4);
    // Kernel-unroll: repeat the whole read workload this many times, no barrier between reps.
    const uint32_t workload_repeat = get_arg_val<uint32_t>(5) > 0 ? get_arg_val<uint32_t>(5) : 1;
    // Mode-2 periodic progress marker: emit a READ_PROG timestamp every N pages completed (0=off).
    const uint32_t read_progress_every = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t READER_MODE = get_compile_time_arg_val(1);
    constexpr uint32_t PUSH_TILE_COUNT = get_compile_time_arg_val(2);
    constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(3);
    constexpr uint32_t TRID_IN_FLIGHT = get_compile_time_arg_val(4);
    constexpr uint32_t CROSS_PROGRAM_OFFSET_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t PROFILE_DETAIL = get_compile_time_arg_val(6);
    // Cheap-read override (reader_mode 2 only): 0 = read the full page; >0 = NoC-read only this
    // many BYTES per page but still push a full CB page (dummy payload -> reads ~free).
    constexpr uint32_t READ_BYTES_OVERRIDE = get_compile_time_arg_val(7);
    constexpr auto src_args = TensorAccessorArgs<8>();

    constexpr uint32_t chunk_size = PUSH_TILE_COUNT > 0 ? PUSH_TILE_COUNT : 1;
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr uint32_t trid_in_flight = TRID_IN_FLIGHT > 0 ? TRID_IN_FLIGHT : 1;

    const auto src = TensorAccessor(src_args, src_addr);

    // Device 2.0 data-movement handles: Noc() binds to this kernel's configured noc_index and
    // CircularBuffer wraps the input CB. All reads/pushes below go through these.
    Noc noc;
    CircularBuffer in_cb(cb_in);

    DETAIL_MARK("PROG_ID", program_id);
    DETAIL_MARK("NCRISC_GO", program_id);

    // Induce read stagger: spin AFTER go (so go stays synchronized) before any reads. 0 = off.
    for (volatile uint32_t d = 0; d < read_start_delay; ++d) {
        asm volatile("nop");
    }

    // When CROSS_PROGRAM_OFFSET_TILES > 0, each program reads a disjoint tile slice
    // so the DRAM controller sees fresh row opens per program (more app-like).
    const uint32_t effective_start_tile_id = start_tile_id + program_id * CROSS_PROGRAM_OFFSET_TILES;
    const uint32_t end_tile_id = effective_start_tile_id + n_tiles;
    const uint32_t tile_bytes = get_local_cb_interface(cb_in).fifo_page_size;
    // Full DRAM page in bytes (the src accessor's page_size). Modes 0/1 force tiles_per_page==1
    // (page == tile); mode 2 reads whole pages of tiles_per_page tiles per NoC transaction.
    const uint32_t page_bytes = tiles_per_page * tile_bytes;

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
        // reserve_back / push_back are kept purely for flow control so the
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

        for (uint32_t rep = 0; rep < workload_repeat; ++rep) {  // kernel-unroll: no barrier between reps
            // Reserve room for both in-flight batches up front (covers prologue + head).
            in_cb.reserve_back(2 * batch_tiles);
            const uint32_t cb_base = in_cb.get_write_ptr();
            // Every read in this rep lands at the fixed cb_base (dummy payload; see above), so wrap
            // it in a CoreLocalMem instead of letting the CB write pointer walk as we push.
            CoreLocalMem<uint32_t> dst(cb_base);

            // Issue one page read into cb_base tagged with `trid`. Normally a full-page tile read; in
            // cheap-read mode only READ_BYTES_OVERRIDE bytes (dummy payload) so the read side is ~free
            // and the writer becomes the bottleneck (output-bound). Device 2.0 async_read with
            // NocOptions::TXN_ID programs the transaction id per issue (replaces noc_async_read_set_trid).
            auto issue_read = [&](uint32_t pid, uint32_t trid) {
                if constexpr (READ_BYTES_OVERRIDE > 0) {
                    noc.async_read<NocOptions::TXN_ID>(
                        src, dst, READ_BYTES_OVERRIDE, {.page_id = pid}, {}, NocOptVals{.trid = trid});
                } else {
                    noc.async_read<NocOptions::TXN_ID>(
                        src, dst, page_bytes, {.page_id = pid}, {}, NocOptVals{.trid = trid});
                }
            };

            DETAIL_MARK("READ_BEFORE_BARRIER", effective_start_tile_id);

            // Prologue: issue batch 0 on TRID_A.
            uint32_t issue_trid = TRID_A;
            uint32_t drain_trid = TRID_A;
            uint32_t page_id = start_page_id;  // strides across banks, one bump per read
            uint32_t issued = n_pages < N ? n_pages : N;
            for (uint32_t i = 0; i < issued; ++i) {
                issue_read(page_id++, issue_trid);
            }
            uint32_t pushed = 0;

            // Head: if a second batch exists, prefetch it on the other trid (so two batches
            // are in flight), then drain batch 0. The first-read marker is emitted here,
            // once, so it never enters the hot loop.
            if (issued < n_pages) {
                issue_trid ^= 1u;
                const uint32_t remaining = n_pages - issued;
                const uint32_t nb = remaining < N ? remaining : N;
                for (uint32_t i = 0; i < nb; ++i) {
                    issue_read(page_id++, issue_trid);
                }
                issued += nb;

                noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = drain_trid});
                DETAIL_MARK("READ_AFTER_BARRIER", effective_start_tile_id);
                in_cb.push_back(batch_tiles);
                pushed += N;
                if (read_progress_every && pushed % read_progress_every == 0) {
                    DeviceTimestampedData("READ_PROG", pushed);  // measured pages completed vs time
                }
                drain_trid ^= 1u;
            }

            // Hot loop: uniform prefetch-then-drain, no data-dependent branch. The reserve
            // gates reuse of the region the oldest batch just vacated.
            while (issued < n_pages) {
                issue_trid ^= 1u;
                const uint32_t remaining = n_pages - issued;
                const uint32_t nb = remaining < N ? remaining : N;
                in_cb.reserve_back(batch_tiles);
                for (uint32_t i = 0; i < nb; ++i) {
                    issue_read(page_id++, issue_trid);
                }
                issued += nb;

                noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = drain_trid});
                in_cb.push_back(batch_tiles);
                pushed += N;
                if (read_progress_every && pushed % read_progress_every == 0) {
                    DeviceTimestampedData("READ_PROG", pushed);  // measured pages completed vs time
                }
                drain_trid ^= 1u;
            }

            // Tail: drain the final in-flight batch (may be partial). The barrier here
            // completes the last outstanding reads, so it is the read-bandwidth end point:
            // BW = bytes_read / (READ_LAST_BARRIER - READ_BEFORE_BARRIER). The interval
            // spans first-read-issued to last-read-complete, so it still pays the pipeline
            // fill/drain latency tax at each end (small vs the steady-state middle).
            noc.async_read_barrier<NocOptions::TXN_ID>(NocOptVals{.trid = drain_trid});
            if (pushed == 0) {
                // n_pages <= N: head never ran, so emit the first-read marker here.
                DETAIL_MARK("READ_AFTER_BARRIER", effective_start_tile_id);
            }
            DETAIL_MARK("READ_LAST_BARRIER", effective_start_tile_id);
            in_cb.push_back((n_pages - pushed) * tiles_per_page);
            if (read_progress_every) {
                DeviceTimestampedData("READ_PROG", n_pages);  // final: all pages completed
            }
        }  // end kernel-unroll rep loop

        DETAIL_MARK("NCRISC_DONE", program_id);
        return;
    }

    // Legacy reader modes (0 = incremental push-1; 1 = batch read+push).
    for (uint32_t rep = 0; rep < workload_repeat; ++rep) {  // kernel-unroll: no barrier between reps
        for (uint32_t tile_id = effective_start_tile_id; tile_id < end_tile_id;) {
            const uint32_t tiles_left = end_tile_id - tile_id;
            const uint32_t chunk = tiles_left < chunk_size ? tiles_left : chunk_size;

            in_cb.reserve_back(chunk);

            if constexpr (READER_MODE == 1) {
                // Batch: issue the whole chunk, then a single barrier, then push. The
                // barrier must be outside the per-tile loop or every read serializes
                // (which would defeat the point of batch mode). Each read lands at a distinct
                // offset within the freshly reserved region (write ptr is fixed until push_back).
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t tid = tile_id + i;
                    if (rep == 0 && tid == effective_start_tile_id) {
                        DETAIL_MARK("READ_BEFORE_BARRIER", tid);
                    }
                    noc.async_read(src, in_cb, tile_bytes, {.page_id = tid}, {.offset_bytes = i * tile_bytes});
                }
                noc.async_read_barrier();
                if (rep == 0 && tile_id == effective_start_tile_id) {
                    DETAIL_MARK("READ_AFTER_BARRIER", tile_id);
                }
                in_cb.push_back(chunk);
            } else {
                for (uint32_t i = 0; i < chunk; ++i) {
                    const uint32_t tid = tile_id + i;
                    if (rep == 0 && tid == effective_start_tile_id) {
                        DETAIL_MARK("READ_BEFORE_BARRIER", tid);
                    }
                    // Incremental: read one tile to the current write ptr (offset 0), barrier, then
                    // push (advances the write ptr) so the next read lands in the next slot.
                    noc.async_read(src, in_cb, tile_bytes, {.page_id = tid}, {.offset_bytes = 0});
                    noc.async_read_barrier();
                    if (rep == 0 && tid == effective_start_tile_id) {
                        DETAIL_MARK("READ_AFTER_BARRIER", tid);
                    }
                    in_cb.push_back(1);
                }
            }

            tile_id += chunk;
        }
    }  // end kernel-unroll rep loop

    DETAIL_MARK("NCRISC_DONE", program_id);
}
