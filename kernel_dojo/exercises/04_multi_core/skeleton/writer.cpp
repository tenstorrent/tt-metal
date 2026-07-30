// SPDX-License-Identifier: Apache-2.0
//
// Exercise 04 — writer kernel.
//
// Write this core's slice: tiles [start_tile, start_tile + n) of the output.

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_tile = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);

    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto dst = TensorAccessor(dst_args, dst_addr);

    // TODO: loop over this core's slice.
    for (uint32_t i = 0; i < n_tiles; i++) {
        // TODO: wait, write page i, barrier, pop.
    }
}
