// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Tensix gate/up weight forwarder of test_drisc_flat_bench.cpp (NCRISC, NOC1), as flat_routed_expert's se10_fwd.cpp
// without the receivers' credits: each CB 0 page holds R blocks of BLK_BYTES; block j goes to receiver j's landing
// slot c % LAND_SLOTS. The page is freed once its writes have left. Writes the end wall clock at TIMING (word 1).
//
// RT: 0 BLK_BYTES, 1 N, 2 R (<= 8), 3 landing address, 4 LAND_SLOTS, 5 timing, 6.. R x (x, y)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "risc_common.h"
#include "flat_bench_barrier.hpp"

void kernel_main() {
    const uint32_t blk = get_arg_val<uint32_t>(0), n = get_arg_val<uint32_t>(1), R = get_arg_val<uint32_t>(2);
    const uint32_t land = get_arg_val<uint32_t>(3), land_slots = get_arg_val<uint32_t>(4);
    const uint32_t timing = get_arg_val<uint32_t>(5);
    constexpr uint32_t cb = 0;
    uint64_t recv[8];
    for (uint32_t j = 0; j < R; ++j) {
        recv[j] = get_noc_addr(get_arg_val<uint32_t>(6 + 2 * j), get_arg_val<uint32_t>(7 + 2 * j), land);
    }
    for (uint32_t c = 0; c < n; ++c) {
        cb_wait_front(cb, 1);
        const uint32_t src = get_read_ptr(cb);
        const uint32_t off = (c % land_slots) * blk;
        for (uint32_t j = 0; j < R; ++j) {
            noc_async_write(src + j * blk, recv[j] + off, blk);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb, 1);
    }
    noc_async_write_barrier();
    flat_bench_stamp(timing, 1, get_timestamp());
}
