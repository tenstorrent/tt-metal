// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Start barrier for the Tensix kernels of test_drisc_flat_bench.cpp (slow dispatch launches cores ~1 ms apart): write
// 1 into 32 B slot `idx` of the `nall`-slot arrival array at (bx, by, arr), then poll the array until every slot is
// set. `scratch` is 32 B aligned local L1 with room for 32 + 32 * nall bytes.

#pragma once

#include "api/dataflow/dataflow_api.h"

inline void flat_bench_barrier(uint32_t bx, uint32_t by, uint32_t arr, uint32_t idx, uint32_t nall, uint32_t scratch) {
    volatile tt_l1_ptr uint32_t* one = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    one[0] = 1;
    noc_async_write(scratch, get_noc_addr(bx, by, arr + 32 * idx), sizeof(uint32_t));
    noc_async_write_barrier();
    volatile tt_l1_ptr uint32_t* seen = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + 32);
    for (bool all = false; !all;) {
        noc_async_read(get_noc_addr(bx, by, arr), scratch + 32, 32 * nall);
        noc_async_read_barrier();
        invalidate_l1_cache();
        all = true;
        for (uint32_t i = 0; i < nall && all; ++i) {
            all = seen[8 * i] != 0;
        }
    }
}

inline void flat_bench_stamp(uint32_t timing, uint32_t word, uint64_t t) {
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(timing);
    out[2 * word] = static_cast<uint32_t>(t);
    out[2 * word + 1] = static_cast<uint32_t>(t >> 32);
}
