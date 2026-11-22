// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill MoE routing, pass 2 (per core, after pass 1 on every core). From all cores' histograms:
//   cnt[e] = sum over cores, off[e] = sum over e' < e of cnt[e'] rounded up to 32 (expert e's buffer region),
//   pair (tok, k) with expert e is buffer row off[e] + (pairs of e on cores < c) + (its rank among this
//   core's pairs of e, pair order tok * K + k) -- the same rows as a stable sort of all pairs by expert.
// Writes back[k, tok] = buffer row of (tok, k) for this core's tokens (tokens >= n: the spare row), the row
// index xri[row] = tok (one contiguous run per expert), and (core 0) off and cnt.
// Runtime args: idx, nreal, hist, off, cnt, xri, back, R0, R1, c, n_cores.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

// L1 scratch starts at 64 bytes (the DRAM alignment), so a DRAM transfer has both ends aligned alike.
static inline uint32_t al64(uint32_t a) { return (a + 63) & ~63u; }

void kernel_main() {
    constexpr uint32_t NE = get_named_compile_time_arg_val("ne");
    constexpr uint32_t K = get_named_compile_time_arg_val("k");
    constexpr uint32_t T = get_named_compile_time_arg_val("t");
    constexpr uint32_t SPARE = get_named_compile_time_arg_val("spare");  // zeroed spare buffer row
    constexpr bool idx_dram = get_named_compile_time_arg_val("idx_dram");
    constexpr uint32_t cb_idx = 0, cb_hist = 1, cb_n = 2, cb_back = 3, cb_xri = 4, cb_tab = 5;
    const uint32_t idx = get_arg_val<uint32_t>(0), nreal = get_arg_val<uint32_t>(1), hist = get_arg_val<uint32_t>(2);
    const uint32_t off = get_arg_val<uint32_t>(3), cnt = get_arg_val<uint32_t>(4), xri = get_arg_val<uint32_t>(5);
    const uint32_t back = get_arg_val<uint32_t>(6), R0 = get_arg_val<uint32_t>(7), R1 = get_arg_val<uint32_t>(8);
    const uint32_t c = get_arg_val<uint32_t>(9), NC = get_arg_val<uint32_t>(10);
    const InterleavedAddrGen<idx_dram> ia = {.bank_base_address = idx, .page_size = 2048};
    const InterleavedAddrGen<true> one_n = {.bank_base_address = nreal, .page_size = 64};
    const InterleavedAddrGen<true> one_h = {.bank_base_address = hist, .page_size = 64};
    const InterleavedAddrGen<true> one_o = {.bank_base_address = off, .page_size = 64};
    const InterleavedAddrGen<true> one_c = {.bank_base_address = cnt, .page_size = 64};
    const InterleavedAddrGen<true> one_x = {.bank_base_address = xri, .page_size = 64};
    const InterleavedAddrGen<true> one_b = {.bank_base_address = back, .page_size = 64};

    const uint32_t nbuf = al64(get_write_ptr(cb_n)), hb = al64(get_write_ptr(cb_hist)),
                   ib = al64(get_write_ptr(cb_idx));
    noc_async_read(one_n.get_noc_addr(0), nbuf, 64);
    noc_async_read(one_h.get_noc_addr(0), hb, NC * NE * 4);
    for (uint32_t R = R0; R < R1; ++R) {
        const uint32_t l = ib + (R - R0) * 1024;
        noc_async_read(ia.get_noc_addr(R), l, 512);
        noc_async_read(ia.get_noc_addr(R) + 1024, l + 512, 512);
    }
    noc_async_read_barrier();
    const uint32_t n = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(nbuf)[0];
    volatile tt_l1_ptr uint32_t* h = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hb);
    // tab: [0, NE) cnt, [NE, 2 NE) off, [2 NE, 3 NE) next buffer row of this core per expert,
    // [3 NE, 4 NE) next local slot per expert (pairs grouped by expert in xs).
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(al64(get_write_ptr(cb_tab)));
    uint32_t run = 0, loc = 0;
    for (uint32_t e = 0; e < NE; ++e) {
        uint32_t tot = 0, before = 0;
        for (uint32_t j = 0; j < NC; ++j) {
            const uint32_t v = h[j * NE + e];
            tot += v;
            before += j < c ? v : 0;
        }
        tab[e] = tot;
        tab[NE + e] = run;
        tab[2 * NE + e] = run + before;
        tab[3 * NE + e] = loc;
        loc += h[c * NE + e];
        run += (tot + 31) & ~31u;
    }
    if (c == 0) {
        noc_async_write(reinterpret_cast<uint32_t>(tab), one_c.get_noc_addr(0), NE * 4);
        noc_async_write(reinterpret_cast<uint32_t>(tab + NE), one_o.get_noc_addr(0), NE * 4);
    }

    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ib);
    volatile tt_l1_ptr uint32_t* bk = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(al64(get_write_ptr(cb_back)));
    // xs: this core's tokens grouped by expert, run of expert e at local slots [loc_e, loc_e + h[c][e])
    volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(al64(get_write_ptr(cb_xri)));
    const uint32_t ntok = (R1 - R0) * 32;
    for (uint32_t i = 0; i < ntok; ++i) {
        const uint32_t tok = R0 * 32 + i;
        volatile tt_l1_ptr uint16_t* row = ids + i * 16;
        for (uint32_t k = 0; k < K; ++k) {
            if (tok < n) {
                const uint32_t e = row[k];
                bk[k * ntok + i] = tab[2 * NE + e]++;
                xs[tab[3 * NE + e]++] = tok;
            } else {
                bk[k * ntok + i] = SPARE;
            }
        }
    }
    for (uint32_t k = 0; k < K; ++k) {  // back is k-major [K, T]: one run of ntok per k (128-byte aligned)
        noc_async_write(
            reinterpret_cast<uint32_t>(bk + k * ntok), one_b.get_noc_addr(0, (k * T + R0 * 32) * 4), ntok * 4);
    }
    // xri runs: expert e's rows [row0_e, row0_e + h[c][e]); the NOC needs the L1 source and the DRAM
    // destination at the same offset within 64 bytes, so each run is copied to a staging slot at that offset.
    volatile tt_l1_ptr uint32_t* stage = xs + ntok * K;  // 64-byte aligned (ntok * K * 4 is a multiple of 1024)
    uint32_t sp = 0;                                     // words
    uint32_t lo = 0;
    for (uint32_t e = 0; e < NE; ++e) {
        const uint32_t len = h[c * NE + e];
        if (len == 0) {
            continue;
        }
        const uint32_t row0 = tab[2 * NE + e] - len;
        const uint32_t sh = row0 & 15;  // words within 64 bytes
        for (uint32_t j = 0; j < len; ++j) {
            stage[sp + sh + j] = xs[lo + j];
        }
        noc_async_write(reinterpret_cast<uint32_t>(stage + sp + sh), one_x.get_noc_addr(0, row0 * 4), len * 4);
        lo += len;
        sp += (sh + len + 15) & ~15u;
    }
    noc_async_write_barrier();
}
