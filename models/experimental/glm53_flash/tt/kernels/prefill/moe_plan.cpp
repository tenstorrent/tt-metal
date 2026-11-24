// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 16 batched prefill (tt/prefill_tp.py, EP MoE): the per-chip ROUTING PLAN of one prefill chunk, exact integer
// bookkeeping on one core (BRISC), so the expert FFN (moe_fused_swiglu, device-resident counts / region starts) runs
// only this chip's experts on only the rows routed to them.
//
// Pair p = j * C + t (top-k slot j of token t, slot-major): ids[p] = its global expert id. Local expert e = ids[p] - lo
// (lo = this chip's first expert, word 0 of a per-chip tensor; e in [0, E) = this chip's).
//   counts[e]   = number of local pairs of expert e                                   ([1, EP] uint32, EP >= E)
//   starts[e]   = row of expert e's region in the dispatch buffer: tile-aligned (moe_fused_swiglu requires region
//                 starts that are multiples of 32), regions in expert order, each padded to a multiple of 32 rows
//   disp[d]     = token whose row fills dispatch row d (pairs of an expert in p order), C for padding rows (the
//                 gather's zero row)                                                     ([1, CAP] uint32)
//   dest[p]     = dispatch row of local pair p, 0 for another chip's pair (masked by the combine)  ([1, N] uint32)
// All tensors ROW_MAJOR, one page each (DRAM interleaved: page 0). CAP >= N + 32 E (every pair local, every region
// padded) is checked on the host. Scratch: CB PLAN_CB of (2 N + CAP + 3 EP + 32) words.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    constexpr uint32_t C = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t CAP = get_compile_time_arg_val(3);
    constexpr uint32_t EP = get_compile_time_arg_val(4);
    const uint32_t ids_addr = get_arg_val<uint32_t>(0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(1);
    const uint32_t disp_addr = get_arg_val<uint32_t>(2);
    const uint32_t dest_addr = get_arg_val<uint32_t>(3);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(4);
    const uint32_t start_addr = get_arg_val<uint32_t>(5);

    const uint32_t base = get_write_ptr(PLAN_CB);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    volatile tt_l1_ptr uint32_t* dest = ids + N;
    volatile tt_l1_ptr uint32_t* disp = dest + N;
    volatile tt_l1_ptr uint32_t* cnt = disp + CAP;
    volatile tt_l1_ptr uint32_t* start = cnt + EP;
    volatile tt_l1_ptr uint32_t* fill = start + EP;
    volatile tt_l1_ptr uint32_t* lo_w = fill + EP;  // 32 words

    const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = N * 4};
    const InterleavedAddrGen<true> lo_g{.bank_base_address = lo_addr, .page_size = 32 * 4};
    noc_async_read(ids_g.get_noc_addr(0), reinterpret_cast<uint32_t>(ids), N * 4);
    noc_async_read(lo_g.get_noc_addr(0), reinterpret_cast<uint32_t>(lo_w), 32 * 4);
    noc_async_read_barrier();
    const uint32_t lo = lo_w[0];

    for (uint32_t e = 0; e < EP; ++e) {
        cnt[e] = 0;
    }
    for (uint32_t p = 0; p < N; ++p) {
        const uint32_t e = ids[p] - lo;  // wraps above E for another chip's expert below lo
        if (e < E) {
            cnt[e] = cnt[e] + 1;
        }
    }
    uint32_t s = 0;
    for (uint32_t e = 0; e < EP; ++e) {
        start[e] = s;
        fill[e] = s;
        s += (cnt[e] + 31u) & ~31u;
    }
    for (uint32_t d = 0; d < CAP; ++d) {
        disp[d] = C;
    }
    uint32_t t = 0;
    for (uint32_t p = 0; p < N; ++p) {
        const uint32_t e = ids[p] - lo;
        if (e < E) {
            const uint32_t d = fill[e];
            fill[e] = d + 1;
            disp[d] = t;
            dest[p] = d;
        } else {
            dest[p] = 0;
        }
        t = (t + 1 == C) ? 0 : t + 1;
    }

    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = CAP * 4};
    const InterleavedAddrGen<true> dest_g{.bank_base_address = dest_addr, .page_size = N * 4};
    const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
    const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
    noc_async_write(reinterpret_cast<uint32_t>(disp), disp_g.get_noc_addr(0), CAP * 4);
    noc_async_write(reinterpret_cast<uint32_t>(dest), dest_g.get_noc_addr(0), N * 4);
    noc_async_write(reinterpret_cast<uint32_t>(cnt), cnt_g.get_noc_addr(0), EP * 4);
    noc_async_write(reinterpret_cast<uint32_t>(start), start_g.get_noc_addr(0), EP * 4);
    noc_async_write_barrier();
}
