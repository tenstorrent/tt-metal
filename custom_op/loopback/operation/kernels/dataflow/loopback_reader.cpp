// SPDX-License-Identifier: Apache-2.0
//
// Loopback reader: reads tiles from DRAM into cb_in (index 0).
// RT args: [src_base, num_tiles]
// CT args: TensorAccessorArgs<0> (interleave/sharding metadata)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_base = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = 0;

    constexpr auto in_args = TensorAccessorArgs<0>();
    const auto in_acc = TensorAccessor(in_args, src_base, get_tile_size(cb_in));

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        const uint32_t l1_dst = get_write_ptr(cb_in);

        noc_async_read_tile(t, in_acc, l1_dst);
        noc_async_read_barrier();

        cb_push_back(cb_in, 1);
    }
}
