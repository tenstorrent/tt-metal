// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// compact_stat_gather micro-benchmark — READER (NoC0).
//
// Holds everything except the combine trivial: x is a RESIDENT L1 shard, so
// `load_block` is pure bookkeeping (one publish, then one re-publish per block),
// and there is no gamma and no output staging.  What remains is exactly the
// reader half of the combine: the gather landing (semaphore) and the stat
// multicast, both lifted verbatim from the op.
//
// The landing CB is NoC-zeroed once at boot on the modes that hand-write only
// part of a landing tile (MODE 2 ships two of four faces, MODE 3 ships one row
// per contributor).  Rows / faces a contributor never writes MUST read as zero
// or they enter the root's reduce as a phantom contributor — the same
// "NoC-zero once, hand-write only the real lanes" contract the op's gamma path
// uses.  POISON_LANDING=1 pre-fills that region with a huge value BEFORE the
// zeroing, which pins the zeroing (a leak turns the result into inf/NaN).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;

constexpr uint32_t MODE_RAW_TILE = 0;
constexpr uint32_t MODE_COLLAPSE_4K = 1;

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
    constexpr uint32_t POISON_LANDING = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t LANDING_SEM_ID = get_compile_time_arg_val(CT + 9);

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t GATHER_PAGES = LANDING_ROWS * BLOCK_ROWS;
    // Only the modes that hand-write a SUBSET of a landing tile need the zero.
    constexpr bool NEEDS_ZERO_LANDING = (MODE != MODE_RAW_TILE) && (MODE != MODE_COLLAPSE_4K);

    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 0);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 1);
    // Row-group rect in VIRTUAL coords + its core count — only the root drives this
    // (the landing-ready broadcast below).
    const uint32_t rect_sx = get_arg_val<uint32_t>(RT + 2);
    const uint32_t rect_sy = get_arg_val<uint32_t>(RT + 3);
    const uint32_t rect_ex = get_arg_val<uint32_t>(RT + 4);
    const uint32_t rect_ey = get_arg_val<uint32_t>(RT + 5);
    const uint32_t rect_cores = get_arg_val<uint32_t>(RT + 6);

    Noc noc;
    CircularBuffer cb_gather_obj(cb_gathered_partials);

    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();

    // ---- landing-buffer zero + the ORDERING EDGE it needs ----
    //
    // The zero and the contributors' gather writes target the SAME L1 on the root
    // but are driven by DIFFERENT cores, so nothing orders them: a contributor
    // whose Sum(x^2) finishes before the root's zero completes has its partial
    // WIPED (measured — a 256 KB zero at s=8 lost contributors and inflated 1/rms
    // by up to 3.5x).  One broadcast semaphore increment after the zero, waited
    // once per kernel by every contributor's writer, is the whole fix.
    if constexpr (NEEDS_ZERO_LANDING) {
        if (is_root) {
            if constexpr (POISON_LANDING) {
                auto* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_gathered_partials));
                const uint32_t words = (GATHER_PAGES * STAT_TILE_BYTES) / 4;
                for (uint32_t i = 0; i < words; ++i) {
                    p[i] = 0x7149F2CAu;  // 1e30f
                }
                // Order the RISC stores BEFORE the NoC-zero that must overwrite
                // them: the zero engine writes L1 directly, so without a fence the
                // poison can land after it and the pin would report a leak that is
                // the harness's, not the kernel's.
                __asm__ __volatile__("fence" ::: "memory");
            }
            noc.async_write_zeros(cb_gather_obj, GATHER_PAGES * STAT_TILE_BYTES);
            noc.write_zeros_l1_barrier();
            Semaphore<> landing_ready(LANDING_SEM_ID);
            // The rect contains the root, and a NoC multicast does not deliver to
            // its own source — so the fan-out is rect_cores - 1 (which is what the
            // write-ack accounting must see) plus one local set.
            if (rect_cores > 1) {
                landing_ready.inc_multicast(noc, rect_sx, rect_sy, rect_ex, rect_ey, 1, rect_cores - 1);
            }
            landing_ready.up(1);
        }
    }

    // x is resident: publish the whole shard once, then keep the window full.
    cb_reserve_back(cb_input_tiles, IN_WAIT_TILES);
    cb_push_back(cb_input_tiles, IN_WAIT_TILES);

    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        if (block > 0) {
            cb_reserve_back(cb_input_tiles, BLOCK_TILES);
            cb_push_back(cb_input_tiles, BLOCK_TILES);
        }

        if (is_root) {
            cb_reserve_back(cb_gathered_partials, GATHER_PAGES);
            gather_progress.wait_min((block + 1) * NUM_HIDDEN_SLICES);
            cb_push_back(cb_gathered_partials, GATHER_PAGES);

            cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
            cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
            sender_pipe.send(get_read_ptr(cb_rms_bcast), get_write_ptr(cb_rms_recip), BLOCK_ROWS * STAT_TILE_BYTES);
            cb_push_back(cb_rms_recip, BLOCK_ROWS);
            cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
        } else {
            cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
            receiver_pipe.receive();
            cb_push_back(cb_rms_recip, BLOCK_ROWS);
        }
    }
}
