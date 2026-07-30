// SPDX-License-Identifier: Apache-2.0
//
// Exercise 05 — compute kernel, blocked.
//
// Process a whole block inside a single DST acquire/commit, so the math/pack
// handshake is paid once per block instead of once per tile.
//
// DST holds 8 tiles in the default half-sync mode, so `block` never exceeds 8.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_a = get_compile_time_arg_val(0);
    constexpr uint32_t cb_b = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t block = get_compile_time_arg_val(3);

    binary_op_init_common(cb_a, cb_b, cb_out);
    add_tiles_init(cb_a, cb_b);

    for (uint32_t i = 0; i < n_tiles; i += block) {
        // TODO: wait for `block` tiles in each input CB.

        // TODO: reserve `block` pages in cb_out.

        // TODO: acquire DST once, then loop t in [0, block) adding CB tile t
        //       of each input into DST slot t. Commit when the block is done.

        // TODO: wait for DST on the pack side, pack all `block` DST slots into
        //       cb_out, then release.

        // TODO: push `block` pages of output, pop `block` from each input.
    }
}
