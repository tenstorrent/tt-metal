// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read micro-benchmark (test_dram_read_bench.py): the weight reader's forwarder (as se10_fwd.cpp, without the
// receivers' credits). Each CB page bench_stream.cpp pushes holds R blocks of BLK_BYTES; block j goes to landing slot
// c % LAND_SLOTS of receiver j. The page is freed once its writes have left the core (noc_async_writes_flushed), so
// the next chunk's writes overlap the previous one's acknowledgements, like se10_fwd's transaction-id depth.
//
// CT: 0 CB, 1 R (<= 8), 2 BLK_BYTES, 3 N_CHUNKS, 4 LAND_SLOTS
// RT: 0 landing address (the same on every receiver), 1.. R receivers' virtual xy (x << 16 | y)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t blk = get_compile_time_arg_val(2);
    constexpr uint32_t n = get_compile_time_arg_val(3);
    constexpr uint32_t land_slots = get_compile_time_arg_val(4);
    static_assert(R >= 1 && R <= 8, "1..8 receivers");

    const uint32_t land = get_arg_val<uint32_t>(0);
    uint64_t recv[R];
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t xy = get_arg_val<uint32_t>(1 + j);
        recv[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, land);
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
}
