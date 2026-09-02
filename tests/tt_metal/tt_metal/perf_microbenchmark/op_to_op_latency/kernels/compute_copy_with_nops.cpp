// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// "NOP math" compute kernel for the op-to-op latency benchmark.
//
// For each of `n_tiles` tiles the kernel:
//   1. waits for one tile in the input CB and for output-CB space,
//   2. UNPACK stamps the first-math marker when tile i compute starts (TRISC_0 only),
//   3. copy_tile + N×TTI_NOP on math TRISC,
//   4. pack_tile pushes to the output CB (pack TRISC),
//   5. PACK stamps the pack-finish marker once at kernel exit (TRISC_2 only), outside the tile loop.
//
// In lean mode (PROFILE_PER_TILE == 0, the CI path) the tile-0 first-math and the pack-finish
// markers are emitted as DeviceTimestampedData("OP2OP-EVENT", <EV_* id>) -- see the note at the
// emission site; op_to_op_postprocess.py maps event ids 12/13 back to the TILE_IDX /
// FINISH_LAST_PUSH names. In detail mode they are named DeviceTimestampedData markers.
//
// Compile-time args:
//   0: input  CB id
//   1: output CB id
//   2: NUM_NOPS_PER_TILE  (tunable; 0 disables the spin)
//   3: PROFILE_PER_TILE   (1 = stamp TILE_IDX + MATH zone every tile, for latency
//                          analysis; 0 = lean mode for bandwidth measurement: stamp
//                          the first-math event for tile 0 only and drop the per-tile MATH
//                          zone so the per-tile profiler writes don't pace the consumer and
//                          back-pressure the reader. Compute cost is then copy + NOPs
//                          only, which is what we want when balancing NOPs vs read BW.)
//
// Runtime args:
//   0: n_tiles
//   1: program_id  (PROG_ID in device CSV; 0 = pre-compile / warmup)
//   2: workload_repeat  (--kernel-unroll; repeat the tile loop this many times inside ONE
//                        invocation with NO barrier between reps; 0/1 = normal single pass)
//
// Do **not** use DeviceZoneScopedMainN in compute kernel_main (breaks TRISC-KERNEL
// marker pairing). Program-level host gaps: --use-realtime-profiler.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_nops_per_tile = get_compile_time_arg_val(2);
    constexpr uint32_t profile_per_tile = get_compile_time_arg_val(3);

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t program_id = get_arg_val<uint32_t>(1);
    // Kernel-unroll (experiment): consume the workload this many times inside ONE invocation,
    // no barrier between reps. Matches the reader/writer unroll so the whole op runs un-synced.
    const uint32_t workload_repeat = get_arg_val<uint32_t>(2) > 0 ? get_arg_val<uint32_t>(2) : 1;

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);

    // Device 2.0 CB handles for the flow-control ops (wait/reserve/push/pop). The tile-copy /
    // pack LLK calls below still take the raw CB ids.
    CircularBuffer in_cb(cb_in);
    CircularBuffer out_cb(cb_out);

    // Lean-mode markers were DeviceRecordEvent (runtime event id, no payload) in the DRAM-backend
    // era, where the id rode in the marker's timer_id field and op_to_op_postprocess.py recovered it
    // as (timer_id & 0xFFFF). That marker type is gone: on the streaming wire a runtime value is
    // ordinary DeviceTimestampedData PAYLOAD (and nothing on this backend is compiled out under
    // TT_METAL_PROFILER_ACCUMULATE, so the old survives-accumulate motivation is moot). The EV_*
    // encoding is preserved verbatim as the payload; op_to_op_postprocess.py must read it from the
    // marker's data word instead of timer_id when this benchmark next runs on the streaming wire.
    constexpr uint16_t EV_UNPACK_TILE0 = 12, EV_PACK_FINISH = 13, EV_PROG_BASE = 64;

    // Program id, encoded exactly as before (EV_PROG_BASE + program_id), now as payload.
    DeviceTimestampedData("OP2OP-EVENT", static_cast<uint16_t>(EV_PROG_BASE + program_id));

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
            in_cb.wait_front(1);
            out_cb.reserve_back(1);

            // Per-tile TILE_IDX marker only in profiling mode; lean mode keeps just tile 0
            // (op-to-op latency needs the program's first-tile compute start).
            if constexpr (profile_per_tile) {
                UNPACK(DeviceTimestampedData("TILE_IDX", i));
            } else {
                if (rep == 0 && i == 0) {
                    UNPACK(DeviceTimestampedData("OP2OP-EVENT", EV_UNPACK_TILE0));  // lean first-math (rep0 tile0)
                }
            }

            if constexpr (profile_per_tile) {
                DeviceZoneScopedN("MATH");
                copy_one_tile();
            } else {
                copy_one_tile();
            }

            out_cb.push_back(1);
            in_cb.pop_front(1);
        }
    }

    // Pack-finish; the owning program is recovered from the PROG_ID marker above.
    PACK(DeviceTimestampedData("OP2OP-EVENT", EV_PACK_FINISH));
}
