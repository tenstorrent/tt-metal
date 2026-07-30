// SPDX-License-Identifier: Apache-2.0
//
// Exercise 02 — compute kernel.
//
// For each of n_tiles tiles: take a tile from cb_in, compute exp() of it on the
// SFPU, and hand the result to cb_out.
//
// Remember: the SFPU works in place on DST registers, so the tile has to be
// copied into DST before exp can touch it.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);

    // We only ever use one DST slot in this exercise.
    constexpr uint32_t dst = 0;

    // TODO: configure the hardware for a unary op reading cb_in, writing cb_out.
    //       (this must be the first compute call in the kernel)

    // TODO: program the SFPU for exponential — once, out here, not in the loop.

    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: wait for one tile in cb_in.

        // TODO: acquire DST for the math thread.

        // TODO: copy tile 0 of cb_in into DST slot `dst`.

        // TODO: apply exp to DST slot `dst`, in place.

        // TODO: commit DST from the math thread.

        // TODO: reserve one page in cb_out.

        // TODO: wait for DST on the pack thread.

        // TODO: pack DST slot `dst` into cb_out.

        // TODO: release DST from the pack thread.

        // TODO: push the output page and pop the consumed input tile.
    }
}
