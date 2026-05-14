// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// "NOP math" compute kernel for the op-to-op latency benchmark.
//
// For each of `n_tiles` tiles the kernel:
//   1. waits for one tile in the input CB and for output-CB space,
//   2. opens the per-tile MATH profiler zone (so the start timestamp marks
//      the moment data is actually available to the tensix and there is
//      space to write the result -- i.e. the earliest point at which
//      real math work can begin),
//   3. issues a real copy_tile so the unpacker / math / packer all run
//      (matches the SFPU/FPU work an actual op would do),
//   4. spins on a tunable, unrolled loop of TTI_NOPs to stretch the kernel
//      runtime to a configurable target (~10us per program is typical).
//      TTI_NOP expands to `__asm__ __volatile__(".ttinsn ...")` -- the
//      __volatile__ qualifier prevents the compiler from removing or
//      reordering the asm statement, so the loop body cannot be optimised
//      out regardless of optimisation level.
//   5. packs the result and pushes the tile to the output CB (after the
//      MATH zone has closed, since cb_push_back / cb_pop_front are
//      dataflow signalling, not math).
//
// Compile-time args:
//   0: input  CB id
//   1: output CB id
//   2: NUM_NOPS_PER_TILE  (tunable; 0 disables the spin)
//
// Runtime args:
//   0: n_tiles  (per-core slice size)
//
// Profiling layout (per-core, per-TRISC, per-program-run):
//
//   * Do **not** use DeviceZoneScopedMainN here: user `kernel_main` runs
//     inside firmware `DeviceZoneScopedMainChildN("TRISC-KERNEL")`
//     (tt_metal/hw/firmware/src/tt-1xx/trisck.cc). MainN uses guaranteed
//     slot index 0 whose destructor calls finish_profiler() while the
//     parent TRISC-KERNEL zone is still open, which breaks host-side
//     start/end pairing (TT_FATAL: marker IDs do not match).
//
//   * One `MATH` zone per tile via DeviceZoneScopedN (rolling buffer).
//
//   * `DeviceTimestampedData("TILE_IDX", i)` next to each MATH zone for
//     iteration index in profile_log_device.csv.
//
// Program-level op-to-op (between enqueued programs): use the host
// `--use-realtime-profiler` path (ProgramRealtimeRecord), not an extra
// outer kernel zone here.
//
// All profiler macros are no-ops when kernels are not JIT-built with
// PROFILE_KERNEL (see TT_METAL_DEVICE_PROFILER=1 in rtoptions); no #ifdef
// gating is required on our side.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t num_nops_per_tile = get_compile_time_arg_val(2);

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    unary_op_init_common(cb_in, cb_out);
    copy_tile_init(cb_in);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

        {
            DeviceZoneScopedN("MATH");
            DeviceTimestampedData("TILE_IDX", i);

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
        }

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
