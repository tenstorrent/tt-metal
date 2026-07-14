// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pops `n_tiles` tiles from the output circular buffer (CB_out) and writes
// them to the corresponding tile slots in an interleaved DRAM buffer. One
// instance of this kernel runs on every Tensix core; each core gets its own
// [start_tile_id, start_tile_id + n_tiles) slice via runtime args.
//
// Synchronization (per kernel review feedback):
//   - noc_async_writes_flushed() so the source L1 slot is safe to recycle on
//     cb_pop_front (write has at least left L1).  Mode 0: flush every page.
//     Mode 1: flush only when output CB back-pressure requires it —
//     batch up to (output_cb_depth / tiles_per_page - 1) writes before flush.
//   - noc_async_write_barrier() ONCE at kernel exit so the next op sees fully
//     committed DRAM writes (op-end safety).
//
// Runtime args:
//   0: dst_addr
//   1: n_tiles
//   2: start_tile_id
//   3: program_id
//
// Compile-time args (see TensorAccessorArgs<8> for dst accessor):
//   0: cb_out
//   1: TILES_PER_PAGE
//   2: READ_ONLY        (1 = skip DRAM writes, just pop from output CB)
//   3: WRITER_FLUSH_MODE (0 = flush every page; 1 = flush on CB back-pressure)
//   4: OUTPUT_CB_DEPTH_TILES
//   5: CROSS_PROGRAM_OFFSET_TILES (0 = every program writes same slice; >0 = program k
//                                  writes pages starting at start_tile_id + k*OFFSET)
//   6: END_BARRIER_MODE   (Batch-8 latency experiment — HW-barrier proposal)
//                          0 = noc_async_write_barrier (DEFAULT; wait for DRAM ACK; current)
//                          1 = noc_async_writes_flushed (just flush local L1; writes left
//                              the source but next op may see in-flight committed-at-NoC)
//                          2 = no barrier, no flush (UNSAFE for correctness in real
//                              workloads; simulates "HW gives us this for free")
//   7: PROFILE_DETAIL     (0 = lean CI path: emit no custom markers; 1 = research: emit
//                          GO/DONE/BARRIER markers for BW/gap decomposition)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
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
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(2);
    const uint32_t program_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t TILES_PER_PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t READ_ONLY = get_compile_time_arg_val(2);
    constexpr uint32_t WRITER_FLUSH_MODE = get_compile_time_arg_val(3);
    constexpr uint32_t OUTPUT_CB_DEPTH_TILES = get_compile_time_arg_val(4);
    constexpr uint32_t CROSS_PROGRAM_OFFSET_TILES = get_compile_time_arg_val(5);
    constexpr uint32_t END_BARRIER_MODE = get_compile_time_arg_val(6);
    constexpr uint32_t PROFILE_DETAIL = get_compile_time_arg_val(7);
    constexpr uint32_t tiles_per_page = TILES_PER_PAGE > 0 ? TILES_PER_PAGE : 1;
    constexpr uint32_t output_cb_depth_tiles = OUTPUT_CB_DEPTH_TILES > 0 ? OUTPUT_CB_DEPTH_TILES : tiles_per_page;
    constexpr uint32_t cb_page_slots = output_cb_depth_tiles / tiles_per_page;
    constexpr uint32_t max_writes_before_flush = cb_page_slots > 1 ? cb_page_slots - 1 : 1;
    constexpr auto dst_args = TensorAccessorArgs<8>();

    const auto dst = TensorAccessor(dst_args, dst_addr);

    DETAIL_MARK("PROG_ID", program_id);
    DETAIL_MARK("BRISC_GO", program_id);

    // When CROSS_PROGRAM_OFFSET_TILES > 0, each program writes a disjoint slice
    // (mirrors reader); host allocates a buffer large enough for all slices.
    const uint32_t effective_start_tile_id = start_tile_id + program_id * CROSS_PROGRAM_OFFSET_TILES;
    // DRAM page = tiles_per_page tiles. Compute pushes 1 CB page (1 tile) at
    // a time; we accumulate tiles_per_page in the CB then issue one
    // noc_async_write_tile that pushes the whole page in a single NoC txn.
    const uint32_t start_page_id = effective_start_tile_id / tiles_per_page;
    const uint32_t n_pages = n_tiles / tiles_per_page;
    const uint32_t end_page_id = start_page_id + n_pages;
    const uint32_t last_page_id = (n_pages > 0) ? (end_page_id - 1) : start_page_id;

    DETAIL_MARK("WRITE_BEFORE_BARRIER", last_page_id);

    uint32_t writes_since_flush = 0;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_wait_front(cb_out, tiles_per_page);
        if constexpr (READ_ONLY == 0) {
            const uint32_t l1_read_addr = get_read_ptr(cb_out);
            noc_async_write_tile(page_id, dst, l1_read_addr);
            writes_since_flush++;
            const bool flush_every_page = (WRITER_FLUSH_MODE == 0);
            const bool flush_on_pressure = (WRITER_FLUSH_MODE != 0) && (writes_since_flush >= max_writes_before_flush);
            if (flush_every_page || flush_on_pressure) {
                noc_async_writes_flushed();
                writes_since_flush = 0;
            } else {
                // Pressure mode skipped the periodic flush this iteration. The NoC only
                // guarantees a write source has departed after noc_async_writes_flushed(),
                // so flush before cb_pop_front returns this L1 slot to the producer --
                // otherwise compute can overwrite an in-flight write source and corrupt
                // the DRAM output when the CB is full.
                noc_async_writes_flushed();
            }
        }
        cb_pop_front(cb_out, tiles_per_page);
    }

    if constexpr (READ_ONLY == 0) {
        if ((WRITER_FLUSH_MODE != 0) && (writes_since_flush > 0)) {
            noc_async_writes_flushed();
        }
    }

    if constexpr (READ_ONLY == 0) {
        if constexpr (END_BARRIER_MODE == 0) {
            noc_async_write_barrier();
        } else if constexpr (END_BARRIER_MODE == 1) {
            noc_async_writes_flushed();
        }
    }
    DETAIL_MARK("WRITE_AFTER_BARRIER", last_page_id);

    DETAIL_MARK("BRISC_DONE", program_id);
}
