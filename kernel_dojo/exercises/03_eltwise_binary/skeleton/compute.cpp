// SPDX-License-Identifier: Apache-2.0
//
// Exercise 03 — compute kernel.
//
// c = a + b, one tile at a time, on the FPU.
//
// Unlike the SFPU, the FPU reads its two operands straight out of circular
// buffers — no copy_tile needed.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    constexpr uint32_t dst = 0;

    // TODO: one-time hardware configuration for a binary op over
    //       (cb_a, cb_b) -> cb_out.

    // TODO: tell the FPU the operation is "add".

    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: wait for one tile in each input CB.

        // TODO: acquire DST, add the two tiles into DST slot `dst`, commit.

        // TODO: reserve an output page, wait for DST on the pack side,
        //       pack, release.

        // TODO: push the output, pop *both* inputs.
    }
}
