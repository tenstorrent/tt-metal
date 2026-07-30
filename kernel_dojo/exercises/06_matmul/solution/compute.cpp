// SPDX-License-Identifier: Apache-2.0
//
// Exercise 06 — matmul compute kernel, reference solution.

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

    // Matmul maps in0 -> SrcB and in1 -> SrcA, the reverse of every other op.
    // Getting this wrong produces wrong numbers, not an error.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_a, cb_b, cb_out);
    matmul_init(cb_a, cb_b);

    for (uint32_t mt = 0; mt < Mt; mt++) {
        for (uint32_t nt = 0; nt < Nt; nt++) {
            // Acquiring zeroes DST, so this is a fresh accumulator.
            tile_regs_acquire();

            for (uint32_t kt = 0; kt < Kt; kt++) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);

                // DST[dst] += A_tile @ B_tile
                matmul_tiles(cb_a, cb_b, 0, 0, dst);

                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }

            tile_regs_commit();

            // Pack once, after the whole K reduction. Packing inside the loop
            // would round each partial sum to bfloat16.
            cb_reserve_back(cb_out, 1);
            tile_regs_wait();
            pack_tile(dst, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, 1);
        }
    }
}
