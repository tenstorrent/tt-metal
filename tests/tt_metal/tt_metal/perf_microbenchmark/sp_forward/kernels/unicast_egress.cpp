// SPDX-License-Identifier: Apache-2.0
// Unicast egress probe: one source repeatedly noc_async_write's a logical stream to a SINGLE receiver
// core's L1 (fanout 1, plain unicast, NOT multicast), on NCRISC/NOC1. chunk-tile packets, depth in flight.
// Egress BW = total_tiles*tile_bytes / kernel_time. Companion to mcast_egress to separate the multicast
// replication cost from the raw per-core NOC1 write egress.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_src = get_compile_time_arg_val(1);
    constexpr uint32_t cb_dst = get_compile_time_arg_val(2);

    const uint32_t dst_x = get_arg_val<uint32_t>(0);
    const uint32_t dst_y = get_arg_val<uint32_t>(1);
    const uint32_t total_tiles = get_arg_val<uint32_t>(2);
    const uint32_t chunk = get_arg_val<uint32_t>(3);
    const uint32_t depth = get_arg_val<uint32_t>(4);

    const uint32_t src = get_write_ptr(cb_src);
    const uint32_t dst = get_write_ptr(cb_dst);
    const uint64_t daddr = get_noc_addr(dst_x, dst_y, dst);

    uint32_t sent = 0, inflight = 0;
    while (sent < total_tiles) {
        uint32_t n = (total_tiles - sent < chunk) ? (total_tiles - sent) : chunk;
        noc_async_write(src, daddr, n * tile_bytes);
        inflight++;
        sent += n;
        if (inflight >= depth) {
            noc_async_write_barrier();
            inflight = 0;
        }
    }
    noc_async_write_barrier();
}
