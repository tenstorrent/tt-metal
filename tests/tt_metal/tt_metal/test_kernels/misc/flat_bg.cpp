// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// A flat_routed_expert call's other traffic, synthetic (test_drisc_flat_bench.cpp; as the Python dram_read_bench's
// bench_bg.cpp):
//   MODE 0, a relay: N steps of reading A bytes of x rows from DRAM (bank s % BANKS, at ADDR + (s / BANKS) * A) and
//     multicasting B bytes (the tilized share) into its gate/up rectangle at LAND.
//   MODE 1, a down core: N steps of writing A bytes of h to its chain successor (at LAND) and B bytes of y rows to
//     DRAM (bank s % BANKS, at ADDR + (s / BANKS) * B).
// Writes are flushed once per step. Start barrier; start / end wall clock at TIMING.
//
// RT: 0 MODE, 1 N, 2 A, 3 B, 4 ADDR, 5 BANKS, 6 scratch, 7 LAND, 8 timing, 9 barrier x, 10 y, 11 array, 12 idx,
//     13 nall, then MODE 0: 14-17 rectangle x0 y0 x1 y1 (virtual), 18 cells; MODE 1: 14-15 successor x, y

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"
#include "flat_bench_barrier.hpp"

void kernel_main() {
    const uint32_t mode = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1);
    const uint32_t a = get_arg_val<uint32_t>(2), b = get_arg_val<uint32_t>(3), addr = get_arg_val<uint32_t>(4);
    const uint32_t banks = get_arg_val<uint32_t>(5), scratch = get_arg_val<uint32_t>(6), land = get_arg_val<uint32_t>(7);
    const uint32_t timing = get_arg_val<uint32_t>(8);
    flat_bench_barrier(
        get_arg_val<uint32_t>(9),
        get_arg_val<uint32_t>(10),
        get_arg_val<uint32_t>(11),
        get_arg_val<uint32_t>(12),
        get_arg_val<uint32_t>(13),
        timing + 32);
    flat_bench_stamp(timing, 0, get_timestamp());
    if (mode == 0) {
        const uint64_t mc = get_noc_multicast_addr(
            get_arg_val<uint32_t>(14), get_arg_val<uint32_t>(15), get_arg_val<uint32_t>(16), get_arg_val<uint32_t>(17), land);
        const uint32_t cells = get_arg_val<uint32_t>(18);
        for (uint32_t s = 0; s < n; ++s) {
            noc_async_read(get_noc_addr_from_bank_id<true>(s % banks, addr + (s / banks) * a), scratch, a);
            noc_async_read_barrier();
            noc_async_write_multicast(scratch, mc, b, cells);
            noc_async_writes_flushed();
        }
    } else {
        const uint64_t succ = get_noc_addr(get_arg_val<uint32_t>(14), get_arg_val<uint32_t>(15), land);
        for (uint32_t s = 0; s < n; ++s) {
            noc_async_write(scratch, succ, a);
            noc_async_write(scratch, get_noc_addr_from_bank_id<true>(s % banks, addr + (s / banks) * b), b);
            noc_async_writes_flushed();
        }
    }
    noc_async_write_barrier();
    flat_bench_stamp(timing, 1, get_timestamp());
}
