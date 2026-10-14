// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tensix gate/up weight reader of test_drisc_flat_bench.cpp (BRISC), as flat_routed_expert's se_reader.cpp: N chunks
// of CHUNK_BYTES from its bank (chunk c at ADDR + c * CHUNK_BYTES), one read per barrier, into CB 0 pages for
// flat_tx_fwd.cpp. Start barrier, then the start wall clock at TIMING (word 0; the forwarder writes the end).
//
// RT: 0 bank, 1 addr, 2 CHUNK_BYTES, 3 N, 4 timing, 5 barrier x, 6 y, 7 array, 8 idx, 9 nall

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"
#include "flat_bench_barrier.hpp"

void kernel_main() {
    const uint32_t bank = get_arg_val<uint32_t>(0), addr = get_arg_val<uint32_t>(1);
    const uint32_t chunk = get_arg_val<uint32_t>(2), n = get_arg_val<uint32_t>(3), timing = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb = 0;
    flat_bench_barrier(
        get_arg_val<uint32_t>(5),
        get_arg_val<uint32_t>(6),
        get_arg_val<uint32_t>(7),
        get_arg_val<uint32_t>(8),
        get_arg_val<uint32_t>(9),
        timing + 32);
    flat_bench_stamp(timing, 0, get_timestamp());
    const uint64_t src = get_noc_addr_from_bank_id<true>(bank, addr);
    for (uint32_t c = 0; c < n; ++c) {
        cb_reserve_back(cb, 1);
        noc_async_read(src + uint64_t(c) * chunk, get_write_ptr(cb), chunk);
        noc_async_read_barrier();
        cb_push_back(cb, 1);
    }
}
