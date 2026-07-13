// SPDX-License-Identifier: Apache-2.0
// EXP1: single-core L1->L1 multicast egress ceiling. Source multicasts `total_tiles` (logical) from an
// L1 scratch buffer to a rectangle of K receiver cores, in `chunk`-tile packets, `depth` writes in
// flight before a barrier. Runs on NCRISC/NOC1 (the idle NoC). Fire-and-forget (data irrelevant).
// Egress BW = total_tiles*tile_bytes / kernel_time (counted ONCE, not x K -- routers replicate).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(2);

    const uint32_t num_dests = get_arg_val<uint32_t>(0);
    const uint32_t x0 = get_arg_val<uint32_t>(1);
    const uint32_t y0 = get_arg_val<uint32_t>(2);
    const uint32_t x1 = get_arg_val<uint32_t>(3);
    const uint32_t y1 = get_arg_val<uint32_t>(4);
    const uint32_t total_tiles = get_arg_val<uint32_t>(5);
    const uint32_t chunk = get_arg_val<uint32_t>(6);
    const uint32_t depth = get_arg_val<uint32_t>(7);

    const uint32_t src = get_write_ptr(cb_src);
    const uint32_t dst = get_write_ptr(cb_dst);
    const uint64_t maddr = get_noc_multicast_addr(x0, y0, x1, y1, dst);

    uint32_t sent = 0;
    uint32_t inflight = 0;
    while (sent < total_tiles) {
        uint32_t n = (total_tiles - sent < chunk) ? (total_tiles - sent) : chunk;
        noc_async_write_multicast(src, maddr, n * tile_bytes, num_dests);
        inflight++;
        sent += n;
        if (inflight >= depth) {
            noc_async_write_barrier();
            inflight = 0;
        }
    }
    noc_async_write_barrier();
}
