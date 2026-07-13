// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// SP5 microbenchmark reader: read a CONTIGUOUS range of tile ids from a DRAM-INTERLEAVED
// tensor (logical-order read, the layout Regime B is forced to use). Outstanding depth is
// controlled by `depth`; two L1 regions are double-buffered so issue of one batch overlaps
// the in-flight completion of the previous batch. Runs on either RISC (cb_id/region set by host).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t depth = get_compile_time_arg_val(0);  // tiles per batch (outstanding)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id = get_compile_time_arg_val(2);
    constexpr auto args = TensorAccessorArgs<3>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);

    const auto s = TensorAccessor(args, input_addr, tile_bytes);

    const uint32_t base = get_write_ptr(cb_id);  // scratch region: 2*depth tiles

    uint32_t tid = start_tile_id;
    uint32_t remaining = num_tiles;
    uint32_t region = 0;
    bool have_prev = false;

    while (remaining > 0) {
        uint32_t n = remaining < depth ? remaining : depth;
        uint32_t l1 = base + region * depth * tile_bytes;
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read_page(tid, s, l1);
            ++tid;
            l1 += tile_bytes;
        }
        if (have_prev) {
            noc_async_read_barrier();
        }
        have_prev = true;
        region ^= 1;
        remaining -= n;
    }
    noc_async_read_barrier();
}
