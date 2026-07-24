// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// "NOP math" compute kernel for the op-to-op latency benchmark.
//
// For each of `n_tiles` tiles the kernel:
//   1. waits for one tile in the input CB and for output-CB space,
//   2. UNPACK stamps TILE_IDX when tile i compute starts (TRISC_0 only),
//   3. copy_tile + N×TTI_NOP on math TRISC,
//   4. pack_tile pushes to the output CB (pack TRISC),
//   5. PACK stamps FINISH_LAST_PUSH once at kernel exit (TRISC_2 only), outside the tile loop.
//
// Compile-time args:
//   0: input  CB id
//   1: output CB id
//   2: NUM_NOPS_PER_TILE  (tunable; 0 disables the spin)
//   3: PROFILE_PER_TILE   (1 = stamp TILE_IDX + MATH zone every tile, for latency
//                          analysis; 0 = lean mode for bandwidth measurement: stamp
//                          TILE_IDX only for tile 0 and drop the per-tile MATH zone so
//                          the per-tile profiler writes don't pace the consumer and
//                          back-pressure the reader. Compute cost is then copy + NOPs
//                          only, which is what we want when balancing NOPs vs read BW.)
//
// Runtime args:
//   0: n_tiles
//   1: program_id  (PROG_ID in device CSV; 0 = pre-compile / warmup)
//
// Do **not** use DeviceZoneScopedMainN in compute kernel_main (breaks TRISC-KERNEL
// marker pairing). Program-level host gaps: --use-realtime-profiler.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_nops_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t profile_per_tile = get_compile_time_arg_val(3);

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t program_id = get_arg_val<uint32_t>(1);
    // Kernel-unroll (experiment): consume the workload this many times inside ONE invocation, no
    // barrier between reps. Matches the reader/writer unroll so the whole op runs "twice" un-synced.
    const uint32_t workload_repeat = get_arg_val<uint32_t>(2) > 0 ? get_arg_val<uint32_t>(2) : 1;

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);

    // op_to_op event ids for DeviceRecordEvent (no data payload = less L1 pressure). Kept in
    // sync with export_op_to_op_profiler_csv.py EVENT_NAMES.
    constexpr uint16_t EV_UNPACK_TILE0 = 12, EV_PACK_FINISH = 13;

    DeviceTimestampedData("PROG_ID", program_id);  // keep: program id used for prog association

    // The actual per-tile consumer work: copy CB_in -> dst regs (+ NOP spin) -> CB_out.
    // Kept identical across profiling modes so lean mode changes only instrumentation.
    auto copy_one_tile = [&]() {
        tile_regs_acquire();
        copy_tile(cb_in, /*tile_index=*/0, /*dst_index=*/0);

#pragma GCC unroll 65534
        for (uint32_t j = 0; j < num_nops_per_tile; ++j) {
            TTI_NOP;
        }

        tile_regs_commit();

        tile_regs_wait();
        pack_tile(/*src_index=*/0, cb_out);
        tile_regs_release();
    };

    for (uint32_t rep = 0; rep < workload_repeat; ++rep) {  // kernel-unroll: no barrier between reps
        for (uint32_t i = 0; i < n_tiles; ++i) {
            cb_wait_front(cb_in, 1);
            cb_reserve_back(cb_out, 1);

            // Per-tile TILE_IDX marker only in profiling mode; lean mode keeps just tile 0
            // (op-to-op latency needs the program's first-tile compute start).
            if constexpr (profile_per_tile) {
                UNPACK(DeviceTimestampedData("TILE_IDX", i));
            } else {
                if (rep == 0 && i == 0) {
                    UNPACK(DeviceRecordEvent(EV_UNPACK_TILE0));  // lean: first-math marker (rep0 tile0)
                }
            }

            if constexpr (profile_per_tile) {
                DeviceZoneScopedN("MATH");
                copy_one_tile();
            } else {
                copy_one_tile();
            }

            cb_push_back(cb_out, 1);
            cb_pop_front(cb_in, 1);
        }
    }

    PACK(DeviceRecordEvent(EV_PACK_FINISH));
}
