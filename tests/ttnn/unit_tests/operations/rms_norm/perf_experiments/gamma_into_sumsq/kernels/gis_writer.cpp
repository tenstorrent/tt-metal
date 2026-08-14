// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH — "where does apply_gamma belong?", writer (NoC1) half.
//
// IDENTICAL in every variant.  Reconstructs the shipped rms_norm writer for the
// TILE + BLOCK-sharded geometry:
//   * the contributor half of the reduce-scatter combine — this core's per-slice
//     partial for row r goes to the core that OWNS row r, page
//     ((r % own_rows)*s + slice_index), plus one unicast atomic per owner;
//   * `store_block` — on a resident output shard cb_output_tiles IS the shard, so
//     compute already packed the tile-row into its final home and the store is the
//     pop that advances the window.
//
// `wr_stat_wait` is the number the whole bake-off turns on: it is how long this
// core's gather sits waiting for its OWN Sum(x^2).  Moving the gamma multiply in
// FRONT of the partials (the `fused` variant) delays it; moving it BEHIND them but
// in front of the combine (`gamma_first`) does not.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

constexpr uint32_t cb_sq_partials = 2;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_output_tiles = 9;

constexpr uint32_t MAX_OWNERS = 64;

void kernel_main() {
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(2);   // s
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(3);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(4);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(5);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(6);

    const uint32_t num_blocks = get_arg_val<uint32_t>(0);
    const uint32_t slice_index = get_arg_val<uint32_t>(1);

    uint32_t owner_x[MAX_OWNERS];
    uint32_t owner_y[MAX_OWNERS];
    if constexpr (NUM_HIDDEN_SLICES > 1) {
        for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
            owner_x[o] = get_arg_val<uint32_t>(2 + 2 * o);
            owner_y[o] = get_arg_val<uint32_t>(3 + 2 * o);
        }
    }

    Noc noc;
    Semaphore<> gather_progress(GATHER_SEM_ID);
    uint32_t gather_base = 0;
    if constexpr (NUM_HIDDEN_SLICES > 1) {
        gather_base = get_write_ptr(cb_gathered_partials);
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if constexpr (NUM_HIDDEN_SLICES > 1) {
            {
                MaybeDeviceZoneScope("wr_stat_wait");
                cb_wait_front(cb_sq_partials, BLOCK_ROWS);
            }
            const uint32_t src = get_read_ptr(cb_sq_partials);
            {
                MaybeDeviceZoneScope("wr_gather_issue");
                for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                    const uint32_t owner = r / OWN_ROWS;
                    const uint32_t page = (r % OWN_ROWS) * NUM_HIDDEN_SLICES + slice_index;
                    noc_async_write(
                        src + r * STAT_TILE_BYTES,
                        get_noc_addr(owner_x[owner], owner_y[owner], gather_base + page * STAT_TILE_BYTES),
                        STAT_TILE_BYTES);
                }
            }
            {
                MaybeDeviceZoneScope("wr_gather_barrier");
                noc_async_write_barrier();
            }
            {
                MaybeDeviceZoneScope("wr_gather_signal");
                for (uint32_t o = 0; o < NUM_OWNERS; ++o) {
                    gather_progress.up(noc, owner_x[o], owner_y[o], 1);
                }
            }
            cb_pop_front(cb_sq_partials, BLOCK_ROWS);
        }

        {
            MaybeDeviceZoneScope("wr_store_total");
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                {
                    MaybeDeviceZoneScope("wr_store_wait");
                    cb_wait_front(cb_output_tiles, SLICE_HIDDEN_TILES);
                }
                // Resident output shard: compute packed straight into it, so the pop
                // IS the store — it moves no bytes.
                cb_pop_front(cb_output_tiles, SLICE_HIDDEN_TILES);
            }
        }
    }
}
