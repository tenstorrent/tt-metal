// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read micro-benchmark (test_dram_read_bench.py): a real call's other traffic, synthetic.
//   MODE 0, a relay (NOC0): N steps of reading READ_BYTES of x rows from DRAM (bank step % 8), then multicasting
//     MCAST_BYTES (the tilized share) into its gate/up rectangle.
//   MODE 1, a down core (NOC1): N steps of writing H_BYTES of h to its chain successor and Y_BYTES of y rows to DRAM
//     (bank step % 8).
// Writes are flushed once per step (the next step's writes overlap the acknowledgements).
//
// CT: 0 MODE, 1 N, 2 READ_BYTES / H_BYTES, 3 MCAST_BYTES / Y_BYTES, 4 SCRATCH_CB
// RT MODE 0: 0 DRAM base, 1 DRAM offset, 2 landing address, 3-6 rectangle virtual x0 y0 x1 y1, 7 cells in it
// RT MODE 1: 0 DRAM base, 1 DRAM offset, 2 landing address, 3 successor virtual xy

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t n = get_compile_time_arg_val(1);
    constexpr uint32_t a_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t b_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t cb = get_compile_time_arg_val(4);

    const uint32_t base = get_arg_val<uint32_t>(0), off = get_arg_val<uint32_t>(1), land = get_arg_val<uint32_t>(2);
    const uint32_t scratch = get_write_ptr(cb);
    if constexpr (mode == 0) {
        const uint64_t mc = get_noc_multicast_addr(
            get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), land);
        const uint32_t cells = get_arg_val<uint32_t>(7);
        for (uint32_t s = 0; s < n; ++s) {
            noc_async_read(get_noc_addr_from_bank_id<true>(s % 8, base + off + (s / 8) * a_bytes), scratch, a_bytes);
            noc_async_read_barrier();
            noc_async_write_multicast(scratch, mc, b_bytes, cells);
            noc_async_writes_flushed();
        }
    } else {
        const uint32_t xy = get_arg_val<uint32_t>(3);
        const uint64_t succ = get_noc_addr(xy >> 16, xy & 0xFFFF, land);
        for (uint32_t s = 0; s < n; ++s) {
            noc_async_write(scratch, succ, a_bytes);
            noc_async_write(scratch, get_noc_addr_from_bank_id<true>(s % 8, base + off + (s / 8) * b_bytes), b_bytes);
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
}
