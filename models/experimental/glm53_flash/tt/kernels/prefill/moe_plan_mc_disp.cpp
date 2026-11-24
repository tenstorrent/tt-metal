// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 multi-core routing plan (GLM53_PF_MOE_PLAN=mc), program 3 of 3: the dispatch table, the inverse of dest.
// Core i owns dispatch rows [i DCHUNK, min((i + 1) DCHUNK, CAP)): it fills them with C (padding rows -> the gather's
// zero row), then streams every pair's (ids, dest) in blocks of BLK and sets disp[dest[p]] = t(p) for each local pair
// that lands in its range (t = p mod C: pair p = j C + t). The same table moe_plan_v2.cpp builds on one core.
// Compile-time: N, C, E, CAP, DCHUNK (a multiple of 16), BLK (a multiple of 16 dividing N). Runtime: ids, lo, dest,
// disp addresses, core index i. Scratch CB PLAN_CB: DCHUNK + 4 BLK + 32 words (two (ids, dest) block buffers).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t CAP = get_compile_time_arg_val(3);
    constexpr uint32_t DCHUNK = get_compile_time_arg_val(4);
    constexpr uint32_t BLK = get_compile_time_arg_val(5);
    static_assert(DCHUNK % 16 == 0 && BLK % 16 == 0 && N % BLK == 0, "64-B aligned ranges");
    constexpr uint32_t NB = N / BLK;
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(1);
    const uint32_t dest_addr = get_arg_val<uint32_t>(2);
    const uint32_t disp_addr = get_arg_val<uint32_t>(3);
    const uint32_t i = get_arg_val<uint32_t>(4);
    const uint32_t d0 = i * DCHUNK;
    if (d0 >= CAP) {
        return;  // no rows: nothing read, nothing in flight
    }
    const uint32_t nd = CAP - d0 < DCHUNK ? CAP - d0 : DCHUNK;

    const uint32_t base_l1 = get_write_ptr(PLAN_CB);
    uint32_t* disp = reinterpret_cast<uint32_t*>(base_l1);
    uint32_t* bufs = disp + DCHUNK;  // [2][ids BLK | dest BLK]
    uint32_t* lo_w = bufs + 4 * BLK;
    const InterleavedAddrGen<true> lo_g{.bank_base_address = lo_addr, .page_size = 32 * 4};
    const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = N * 4};
    const InterleavedAddrGen<true> dest_g{.bank_base_address = dest_addr, .page_size = N * 4};
    auto issue = [&](uint32_t b) {
        uint32_t* buf = bufs + (b & 1) * 2 * BLK;
        noc_async_read(ids_g.get_noc_addr(0, b * BLK * 4), reinterpret_cast<uint32_t>(buf), BLK * 4);
        noc_async_read(dest_g.get_noc_addr(0, b * BLK * 4), reinterpret_cast<uint32_t>(buf + BLK), BLK * 4);
    };
    noc_async_read(lo_g.get_noc_addr(0), reinterpret_cast<uint32_t>(lo_w), 32 * 4);
    issue(0);
    for (uint32_t d = 0; d < nd; ++d) {
        disp[d] = C;
    }
    noc_async_read_barrier();
    const uint32_t lo = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lo_w);

    uint32_t t = 0;
    for (uint32_t b = 0; b < NB; ++b) {
        if (b + 1 < NB) {
            issue(b + 1);  // the next block streams in while this one is scanned
        }
        const uint32_t* ids = bufs + (b & 1) * 2 * BLK;
        const uint32_t* dst = ids + BLK;
        for (uint32_t k = 0; k < BLK; ++k) {
            const uint32_t e = ids[k] - lo;
            const uint32_t r = dst[k] - d0;  // wraps above nd for rows below d0
            if (e < E && r < nd) {
                disp[r] = t;
            }
            t = (t + 1 == C) ? 0 : t + 1;
        }
        noc_async_read_barrier();
    }
    asm volatile("fence" ::: "memory");  // the table is in L1 before the NoC reads it
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = CAP * 4};
    noc_async_write(reinterpret_cast<uint32_t>(disp), disp_g.get_noc_addr(0, d0 * 4), nd * 4);
    noc_async_write_barrier();
}
