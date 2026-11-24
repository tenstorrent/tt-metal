// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 multi-core routing plan (tt/prefill_tp.py EpMoe, GLM53_PF_MOE_PLAN=mc), program 1 of 3: each core counts
// the local experts of its pair chunk [i CHUNK, min((i + 1) CHUNK, N)) into row i of hist ([P, EP] uint32 ROW_MAJOR,
// one page per core). Same pair / expert conventions as moe_plan_v2.cpp (pair p = j C + t, e = ids[p] - lo, local iff
// e < E). Compile-time: N, E, EP, CHUNK (a multiple of 16: 64-B aligned DRAM reads). Runtime: ids, lo, hist addresses,
// core index i. Scratch CB PLAN_CB: CHUNK + EP + 32 words.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t E = get_compile_time_arg_val(1);
    constexpr uint32_t EP = get_compile_time_arg_val(2);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(3);
    static_assert(CHUNK % 16 == 0 && EP % 16 == 0, "64-B aligned chunks / rows");
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(1);
    const uint32_t hist_addr = get_arg_val<uint32_t>(2);
    const uint32_t i = get_arg_val<uint32_t>(3);

    const uint32_t base = get_write_ptr(PLAN_CB);
    uint32_t* ids = reinterpret_cast<uint32_t*>(base);
    uint32_t* cnt = ids + CHUNK;
    uint32_t* lo_w = cnt + EP;  // 32 words
    const uint32_t p0 = i * CHUNK;
    const uint32_t n = p0 >= N ? 0 : (N - p0 < CHUNK ? N - p0 : CHUNK);

    const InterleavedAddrGen<true> lo_g{.bank_base_address = lo_addr, .page_size = 32 * 4};
    noc_async_read(lo_g.get_noc_addr(0), reinterpret_cast<uint32_t>(lo_w), 32 * 4);
    if (n > 0) {
        const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = N * 4};
        noc_async_read(ids_g.get_noc_addr(0, p0 * 4), reinterpret_cast<uint32_t>(ids), (n * 4 + 63) / 64 * 64);
    }
    noc_async_read_barrier();
    const uint32_t lo = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lo_w);
    for (uint32_t e = 0; e < EP; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t p = 0; p < n; ++p) {
        const uint32_t e = ids[p] - lo;  // wraps above E for another chip's expert below lo
        if (e < E) {
            cnt[e] += 1;
        }
    }
    asm volatile("fence" ::: "memory");  // the counts are in L1 before the NoC reads them
    const InterleavedAddrGen<true> hist_g{.bank_base_address = hist_addr, .page_size = EP * 4};
    noc_async_write(reinterpret_cast<uint32_t>(cnt), hist_g.get_noc_addr(i), EP * 4);
    noc_async_write_barrier();
}
