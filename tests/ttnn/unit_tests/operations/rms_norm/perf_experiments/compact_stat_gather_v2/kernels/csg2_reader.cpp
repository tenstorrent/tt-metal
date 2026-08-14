// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather_v2 micro-benchmark — READER (NoC0).
//
// Reconstructs the reader half of rms_norm's POST-Perf-1 combine: the per-owner
// gather landing, the owner -> root funnel, and the root's stat broadcast.
// Everything else is held trivial: x is a RESIDENT L1 shard (load_block is pure
// bookkeeping), there is no gamma and no output staging, so the measured delta
// between MODEs is the gather + owner combine alone.
//
// The landing buffer's un-owned lanes are zeroed by the WRITER, not here -- see
// the long note at the zeroing site below and in csg2_writer.cpp.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_slice_stat = 3;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

constexpr uint32_t MODE_RAW_4K = 0;
constexpr uint32_t MODE_ROW_128B = 1;
constexpr uint32_t MODE_COLLAPSE_2K = 2;
constexpr uint32_t MODE_ROW_64B_PROBE = 3;

void kernel_main() {
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(CT + 0);
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(CT + 1);
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(CT + 2);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t MODE = get_compile_time_arg_val(CT + 6);
    constexpr uint32_t LANDING_ROWS = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t STAT_READY_SEM_ID = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t NUM_OWNERS = get_compile_time_arg_val(CT + 9);
    constexpr uint32_t OWN_ROWS = get_compile_time_arg_val(CT + 10);

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t GATHER_PAGES =
        (MODE == MODE_ROW_128B || MODE == MODE_ROW_64B_PROBE) ? (LANDING_ROWS * OWN_ROWS)
                                                              : (NUM_HIDDEN_SLICES * OWN_ROWS);

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    const uint32_t is_owner = get_arg_val<uint32_t>(RT + 2);
    const uint32_t my_first_row = get_arg_val<uint32_t>(RT + 3);
    const uint32_t root_noc_x = get_arg_val<uint32_t>(RT + 4);
    const uint32_t root_noc_y = get_arg_val<uint32_t>(RT + 5);
    const uint32_t rect_sx = get_arg_val<uint32_t>(RT + 6);
    const uint32_t rect_sy = get_arg_val<uint32_t>(RT + 7);
    const uint32_t rect_ex = get_arg_val<uint32_t>(RT + 8);
    const uint32_t rect_ey = get_arg_val<uint32_t>(RT + 9);
    const uint32_t rect_cores = get_arg_val<uint32_t>(RT + 10);

    Noc noc;

    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    // NOTE: the landing buffer's un-owned lanes are NOT zeroed here.
    //
    // Round 1 zeroed them on the (single) root and paid for it with a cross-core
    // ORDERING EDGE — a boot semaphore every contributor had to wait on, because
    // the zero and the contributors' writes hit the same L1 from different cores
    // (unordered; the bug it prevents was measured at stat PCC 0.048).  Under the
    // reduce-scatter topology that edge has NUM_OWNERS senders instead of one, and
    // it HANGS: measured twice here, once with `inc_multicast` (s=2/B=8, both cores
    // owners) and once with per-core unicast atomics (the focus geometry, 3 of 8
    // row-groups stuck with 4 writers parked in `landing_ready.wait_min`).
    //
    // The edge is now gone BY CONSTRUCTION: every byte of the landing buffer has
    // exactly ONE writer (see csg2_writer.cpp's boot pad-zero), and that writer's
    // own `noc_async_write_barrier` + `gather_progress` increment is already the
    // edge the owner's reduce waits on.  No semaphore, no zero engine, no race.

    // x is resident: publish the whole shard once, then keep the window full.
    cb_reserve_back(cb_input_tiles, IN_WAIT_TILES);
    cb_push_back(cb_input_tiles, IN_WAIT_TILES);

    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);
    Semaphore<> stat_ready(STAT_READY_SEM_ID);

    // cb_rms_bcast's BASE.  Nothing pushes it when the combine is scattered (the
    // owners NoC-write into it), so the pointer stays at base for the kernel's
    // life -- and every CB is declared on one common core set, so the base is the
    // same address on the root as it is here.
    uint32_t rms_bcast_base = 0;
    if constexpr (NUM_OWNERS > 1) {
        rms_bcast_base = get_read_ptr(cb_rms_bcast);
    }

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if (block > 0) {
            cb_reserve_back(cb_input_tiles, BLOCK_TILES);
            cb_push_back(cb_input_tiles, BLOCK_TILES);
        }

        // ---- gather landing (OWNERS) ----
        if (is_owner) {
            cb_reserve_back(cb_gathered_partials, GATHER_PAGES);
            gather_progress.wait_min((block + 1) * NUM_HIDDEN_SLICES);
            cb_push_back(cb_gathered_partials, GATHER_PAGES);
        }

        // ---- the funnel: each owner's OWN_ROWS finalized tiles -> the root ----
        if constexpr (NUM_OWNERS > 1) {
            if (is_owner) {
                cb_wait_front(cb_slice_stat, OWN_ROWS);
                noc_async_write(
                    get_read_ptr(cb_slice_stat),
                    get_noc_addr(root_noc_x, root_noc_y, rms_bcast_base + my_first_row * STAT_TILE_BYTES),
                    OWN_ROWS * STAT_TILE_BYTES);
                noc_async_write_barrier();
                stat_ready.up(noc, root_noc_x, root_noc_y, 1);
                cb_pop_front(cb_slice_stat, OWN_ROWS);
            }
        }

        if (is_root) {
            uint32_t bcast_src;
            if constexpr (NUM_OWNERS > 1) {
                stat_ready.wait_min((block + 1) * NUM_OWNERS);
                bcast_src = rms_bcast_base;
            } else {
                cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
                bcast_src = get_read_ptr(cb_rms_bcast);
            }
            cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
            sender_pipe.send(bcast_src, get_write_ptr(cb_rms_recip), BLOCK_ROWS * STAT_TILE_BYTES);
            cb_push_back(cb_rms_recip, BLOCK_ROWS);
            if constexpr (NUM_OWNERS == 1) {
                cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
            }
        } else {
            cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
            receiver_pipe.receive();
            cb_push_back(cb_rms_recip, BLOCK_ROWS);
        }
    }
}
