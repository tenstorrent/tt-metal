// SPDX-License-Identifier: Apache-2.0
//
// Exercise 06 — matmul compute kernel.
//
// One output tile at a time: accumulate Kt tile-matmuls into a single DST slot,
// then pack once.
//
// Two things differ from every other compute kernel so far:
//   1. startup uses SrcOrder::Reverse — matmul maps in0 to SrcB, in1 to SrcA
//   2. matmul_tiles accumulates (DST += A@B) instead of overwriting

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"

void kernel_main() {
    const uint32_t Mt = get_arg_val<uint32_t>(0);
    const uint32_t Kt = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    constexpr uint32_t dst = 0;

    // TODO: hardware startup for matmul. Mind the SrcOrder.

    // TODO: matmul_init.

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // TODO: acquire DST — this also zeroes it, giving you a clean
            //       accumulator for this output tile.

            for (uint32_t kt = 0; kt < Kt; kt++) {
                // TODO: wait for one tile in each input CB, matmul-accumulate
                //       into DST slot `dst`, then pop both.
            }

            // TODO: commit, reserve an output page, wait, pack once, push,
            //       release. Packing must happen *outside* the kt loop, or the
            //       partial sums get rounded to bfloat16 on every step.
        }
    }
}
