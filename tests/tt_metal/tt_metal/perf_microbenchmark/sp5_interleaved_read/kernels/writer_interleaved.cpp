// SPDX-License-Identifier: Apache-2.0
// SP4: interleaved output write on NCRISC, concurrent with the BRISC big read. Writes num_tiles
// tiles to a DRAM-interleaved output buffer from a fixed L1 scratch (data irrelevant for timing).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t depth = get_compile_time_arg_val(2);
    constexpr uint32_t cb_src = get_compile_time_arg_val(3);
    constexpr auto args = TensorAccessorArgs<4>();

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile = get_arg_val<uint32_t>(1);

    const auto s = TensorAccessor(args, out_addr, tile_bytes);
    const uint32_t src = get_write_ptr(cb_src);

    uint32_t tid = start_tile;
    uint32_t rem = num_tiles;
    bool prev = false;
    while (rem > 0) {
        uint32_t n = rem < depth ? rem : depth;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_write_page(tid, s, src);
            ++tid;
        }
        if (prev) {
            noc_async_write_barrier();
        }
        prev = true;
        rem -= n;
    }
    noc_async_write_barrier();
}
