// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// overlap_combine bench — writer (NoC1).
//
// The contributor half of the combine: this core's STAT_ROWS partial stat tiles
// -> the row-group root's landing pages, plus one progress increment.  Identical
// for every variant except GATHER_DEPTH, which selects which HALF of a
// double-buffered landing buffer this stat block lands in.  That windowing is the
// only dataflow change the pipelined schedule needs.
//
// `store_block` is degenerate here (the output shard is resident L1 and compute
// packs straight into it), so the drain loop at the end moves no bytes; it exists
// so the output CB's lifecycle is closed exactly as the op closes it.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_output_tiles = 9;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t STAT_ROWS = get_compile_time_arg_val(1);           // SB
    constexpr uint32_t NUM_SLICES = get_compile_time_arg_val(2);          // s
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(4);
    constexpr uint32_t GATHER_DEPTH = get_compile_time_arg_val(5);
    constexpr uint32_t SHARD_ROWS = get_compile_time_arg_val(6);

    const uint32_t num_stat_blocks = get_arg_val<uint32_t>(0);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(1);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(2);
    const uint32_t slice_index = get_arg_val<uint32_t>(3);

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM_ID);

    // Captured before any push/pop touches the CB, so this is its BASE — identical
    // on every core of the row-group rect (all CBs are declared on one core set).
    const uint32_t gather_base = get_write_ptr(cb_gathered_partials);

    constexpr uint32_t GATHER_WINDOW_PAGES = NUM_SLICES * STAT_ROWS;

    for (uint32_t sb = 0; sb < num_stat_blocks; ++sb) {
        {
            MaybeDeviceZoneScope("wr_stat_wait");
            cb_wait_front(cb_sq_partials, STAT_ROWS);
        }
        const uint32_t src = get_read_ptr(cb_sq_partials);
        // Which landing half this stat block owns.  Depth 1 => always 0 (the
        // baseline); depth 2 => alternate, which is exactly the run-ahead credit:
        // block sb+2 reuses block sb's pages, and a contributor cannot reach sb+2
        // until it has RECEIVED rms(sb), which the root sends only after its reduce
        // popped sb's pages.  So the broadcast itself is the credit — no extra
        // reverse semaphore is needed.
        const uint32_t win = (GATHER_DEPTH == 1) ? 0u : (sb % GATHER_DEPTH) * GATHER_WINDOW_PAGES;
        {
            MaybeDeviceZoneScope("wr_gather_issue");
            for (uint32_t r = 0; r < STAT_ROWS; ++r) {
                const uint32_t page = win + r * NUM_SLICES + slice_index;
                noc_async_write(
                    src + r * STAT_TILE_BYTES,
                    get_noc_addr(root_noc_x, root_noc_y, gather_base + page * STAT_TILE_BYTES),
                    STAT_TILE_BYTES);
            }
        }
        {
            MaybeDeviceZoneScope("wr_gather_barrier");
            noc_async_write_barrier();
        }
        gather_progress.up(noc, root_noc_x, root_noc_y, 1);
        cb_pop_front(cb_sq_partials, STAT_ROWS);
    }

    // store_block: the output shard is resident, so this only closes the CB.
    for (uint32_t r = 0; r < SHARD_ROWS; ++r) {
        {
            MaybeDeviceZoneScope("wr_store_wait");
            cb_wait_front(cb_output_tiles, SLICE_HIDDEN_TILES);
        }
        cb_pop_front(cb_output_tiles, SLICE_HIDDEN_TILES);
    }
}
