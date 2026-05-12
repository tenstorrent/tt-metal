// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// "NOP math" compute kernel for the op-to-op latency benchmark.
//
// For each of `n_tiles` tiles the kernel:
//   1. waits for one tile in the input CB,
//   2. issues a real copy_tile so the unpacker / math / packer all run
//      (matches the SFPU/FPU work an actual op would do),
//   3. spins on a tunable, unrolled loop of TTI_NOPs to stretch the kernel
//      runtime to a configurable target (Almeet wants ~10us per program).
//      TTI_NOP is inline assembly so the compiler can not optimise it away,
//   4. packs the result and pushes the tile to the output CB.
//
// Compile-time args:
//   0: input  CB id
//   1: output CB id
//   2: NUM_NOPS_PER_TILE  (tunable; 0 disables the spin)
//
// Runtime args:
//   0: n_tiles  (per-core slice size)
//
// Step 4: each iteration of the per-tile loop is wrapped in
// DeviceZoneScopedMainN("MATH"). This emits a "MATH_begin" hardware-cycle
// timestamp on entry and a "MATH_end" timestamp on exit, on every TRISC
// (UNPACK / MATH / PACK), per tile, per program-run, per core. Combined with
// running N back-to-back programs (FD or trace mode), the host can read the
// CSV via ReadMeshDeviceProfilerResults and compute:
//   op_to_op_latency_per_core = MATH_begin[program N+1] - MATH_end[program N]
// using either the MATH-trisc or PACK-trisc rows of the CSV depending on
// what "math finished" means for the analysis.
//
// The macro is a no-op when the binary is not built with --enable-profiler;
// no #ifdef gating is required on our side.

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
        DeviceZoneScopedMainN("MATH");

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_out, 1);

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

        cb_push_back(cb_out, 1);
        cb_pop_front(cb_in, 1);
    }
}
