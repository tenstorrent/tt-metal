// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Multicast bandwidth probe: one core multicasts TOTAL bytes from L1 in PIECE-byte writes to NUM_RECTS rectangles,
// flushing every FLUSH_EVERY pieces (0 = only at the end).
// CT: 0 TOTAL, 1 PIECE, 2 FLUSH_EVERY
// RT: 0 src/dst address, 1 NUM_RECTS, then per rectangle start xy, end xy, destinations
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t total = get_compile_time_arg_val(0);
    constexpr uint32_t piece = get_compile_time_arg_val(1);
    constexpr uint32_t flush_every = get_compile_time_arg_val(2);
    const uint32_t addr = get_arg_val<uint32_t>(0);
    const uint32_t num_rects = get_arg_val<uint32_t>(1);
    uint64_t rect[4];
    uint32_t dests[4];
    for (uint32_t r = 0; r < num_rects; ++r) {
        const uint32_t a0 = get_arg_val<uint32_t>(2 + r * 3), a1 = get_arg_val<uint32_t>(3 + r * 3);
        rect[r] = get_noc_multicast_addr(a0 >> 16, a0 & 0xFFFF, a1 >> 16, a1 & 0xFFFF, 0);
        dests[r] = get_arg_val<uint32_t>(4 + r * 3);
    }
    uint32_t n = 0;
    for (uint32_t off = 0; off < total; off += piece) {
        for (uint32_t r = 0; r < num_rects; ++r) {
            noc_async_write_multicast(addr + (off % 262144), rect[r] | (addr + (off % 262144)), piece, dests[r]);
        }
        if (flush_every && ++n % flush_every == 0) {
            noc_async_write_barrier();
        }
    }
    noc_async_write_barrier();
}
