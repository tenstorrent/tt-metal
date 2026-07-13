// SPDX-License-Identifier: Apache-2.0
// SP3: multicast the small operand from one source core to the compute-core rectangle, on NCRISC,
// concurrent with the BRISC big read. Fire-and-forget from an L1 scratch (data irrelevant for
// timing); measures the NoC-contention cost of the broadcast on the big read.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mc_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t reps = get_compile_time_arg_val(2);
    constexpr uint32_t chunk = get_compile_time_arg_val(3);
    constexpr uint32_t cb_src = get_compile_time_arg_val(4);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(5);

    const uint32_t num_dests = get_arg_val<uint32_t>(0);
    const uint32_t x0 = get_arg_val<uint32_t>(1);
    const uint32_t y0 = get_arg_val<uint32_t>(2);
    const uint32_t x1 = get_arg_val<uint32_t>(3);
    const uint32_t y1 = get_arg_val<uint32_t>(4);

    const uint32_t src = get_write_ptr(cb_src);
    const uint32_t dst = get_write_ptr(cb_dst);

    for (uint32_t r = 0; r < reps; ++r) {
        uint32_t rem = mc_tiles;
        while (rem > 0) {
            uint32_t n = rem < chunk ? rem : chunk;
            uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, dst);
            noc_async_write_multicast(src, maddr, n * tile_bytes, num_dests);
            noc_async_write_barrier();
            rem -= n;
        }
    }
}
