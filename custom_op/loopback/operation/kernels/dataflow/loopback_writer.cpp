// SPDX-License-Identifier: Apache-2.0
//
// Loopback writer: writes tiles from cb_out (index 16) to DRAM.
// RT args: [dst_base, num_tiles]
// CT args: TensorAccessorArgs<0> (interleave/sharding metadata)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_base = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 16;

    constexpr auto out_args = TensorAccessorArgs<0>();
    const auto out_acc = TensorAccessor(out_args, dst_base, get_tile_size(cb_out));

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_wait_front(cb_out, 1);
        const uint32_t l1_src = get_read_ptr(cb_out);

        noc_async_write_tile(t, out_acc, l1_src);
        noc_async_write_barrier();

        cb_pop_front(cb_out, 1);
    }
}
