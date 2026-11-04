// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_plan_mc_dest.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 multi-core routing plan (GLM53_PF_MOE_PLAN=mc), program 2 of 3: every core reads all P histogram rows
// (moe_plan_mc_count.cpp) and derives the plan's region layout exactly as moe_plan_v2.cpp does:
//   counts[e] = sum_i hist[i][e]; starts[e] = tile-aligned exclusive prefix (regions in expert order, each padded to
//   a multiple of 32 rows); rows used = the end of the last region.
// Core i's pairs go to their expert's region after every earlier chunk's pairs of that expert (base[e] = starts[e] +
// sum_{i' < i} hist[i'][e]), in p order -- the single-core plan's fill order -- so dest[p] for its chunk is written as
// one contiguous run (0 for another chip's pair). Core 0 also writes counts, starts and meta (word 0 = rows used, words
// 1..31 = 0).
// SORTED (define PLAN_SORTED, the runs disp program moe_plan_mc_runs.cpp): the chunk's local pairs' tokens are also
// written in (expert, p) order to row i of a staging buffer ([P, CHUNK] uint32): expert e's tokens of this chunk at
// [loff[e], loff[e] + hist[i][e]), loff = the exclusive prefix of hist[i] over the experts.
// Compile-time: N, E, EP, CHUNK, P, C. Runtime: ids, lo, hist, dest, cnt, start, meta addresses, core index i, staging.
// Scratch CB CB_PLAN: P EP + 3 CHUNK + 4 EP + 32 words.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void phase_main() {
    constexpr uint32_t N = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t E = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t EP = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t P = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t C = get_compile_time_arg_val(CT0 + 5);
    static_assert(CHUNK % 16 == 0 && EP % 16 == 0, "64-B aligned chunks / rows");
    const uint32_t ids_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t lo_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t hist_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t dest_addr = get_arg_val<uint32_t>(RT0 + 3);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(RT0 + 4);
    const uint32_t start_addr = get_arg_val<uint32_t>(RT0 + 5);
    const uint32_t meta_addr = get_arg_val<uint32_t>(RT0 + 6);
    const uint32_t i = get_arg_val<uint32_t>(RT0 + 7);
    const uint32_t stage_addr = get_arg_val<uint32_t>(RT0 + 8);

    const uint32_t base_l1 = get_write_ptr(CB_PLAN);
    uint32_t* hist = reinterpret_cast<uint32_t*>(base_l1);  // [P][EP]
    uint32_t* ids = hist + P * EP;
    uint32_t* dest = ids + CHUNK;
    uint32_t* cnt = dest + CHUNK;
    uint32_t* start = cnt + EP;
    uint32_t* fill = start + EP;
    uint32_t* meta = fill + EP;        // 32 words; word 0 = lo on read, rows used on write
    uint32_t* sorted = meta + 32;      // CHUNK words
    uint32_t* lfill = sorted + CHUNK;  // EP words
    const uint32_t p0 = i * CHUNK;
    const uint32_t n = p0 >= N ? 0 : (N - p0 < CHUNK ? N - p0 : CHUNK);

    const InterleavedAddrGen<true> hist_g{.bank_base_address = hist_addr, .page_size = EP * 4};
    for (uint32_t r = 0; r < P; ++r) {
        noc_async_read(hist_g.get_noc_addr(r), reinterpret_cast<uint32_t>(hist + r * EP), EP * 4);
    }
    const InterleavedAddrGen<true> lo_g{.bank_base_address = lo_addr, .page_size = 32 * 4};
    noc_async_read(lo_g.get_noc_addr(0), reinterpret_cast<uint32_t>(meta), 32 * 4);
    if (n > 0) {
#ifdef PMK_IDS_U16
        // v2 (lever 5): ids from the family's router tail = [C, K] uint16 ROW_MAJOR (16 B per token row); this core's
        // slot-major pairs [p0, p0 + n) are rows t0 .. of slot j0 (and the next slot when the chunk wraps): the rows
        // are read as whole 64-B groups (CHUNK is a multiple of 16: p0 % 16 == 0 -> t0 % 16 == 0) and expanded
        {
            constexpr uint32_t K_ = PMK_IDS_K;
            constexpr uint32_t C_ = N / K_;
            const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = C_ * 2 * K_};
            uint32_t* raw = lfill + EP;  // scratch past every table of this phase: up to CHUNK rows x 16 B (+64)
            uint32_t p = 0;
            while (p < n) {
                const uint32_t j = (p0 + p) / C_, t0 = (p0 + p) % C_;
                const uint32_t cnt_ = (C_ - t0 < n - p) ? C_ - t0 : n - p;
                noc_async_read(
                    ids_g.get_noc_addr(0, t0 * 2 * K_),
                    reinterpret_cast<uint32_t>(raw),
                    (cnt_ * 2 * K_ + 63) / 64 * 64);
                noc_async_read_barrier();
                volatile tt_l1_ptr uint16_t* r16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(raw);
                for (uint32_t q = 0; q < cnt_; ++q) {
                    ids[p + q] = r16[q * K_ + j];
                }
                p += cnt_;
            }
        }
#else
        const InterleavedAddrGen<true> ids_g{.bank_base_address = ids_addr, .page_size = N * 4};
        noc_async_read(ids_g.get_noc_addr(0, p0 * 4), reinterpret_cast<uint32_t>(ids), (n * 4 + 63) / 64 * 64);
#endif
    }
    noc_async_read_barrier();
    const uint32_t lo = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);

    uint32_t s = 0;
    for (uint32_t e = 0; e < EP; ++e) {
        uint32_t tot = 0;
        uint32_t before = 0;
        for (uint32_t r = 0; r < P; ++r) {
            const uint32_t h = hist[r * EP + e];
            tot += h;
            before += r < i ? h : 0;
        }
        cnt[e] = tot;
        start[e] = s;
        fill[e] = s + before;
        s += (tot + 31u) & ~31u;
    }
    for (uint32_t p = 0; p < n; ++p) {
        const uint32_t e = ids[p] - lo;
        if (e < E) {
            dest[p] = fill[e];
            fill[e] += 1;
        } else {
#ifdef PMK_EP_SENTINEL  // W7 step 5 (G, in-program EP): another chip's pair -> the sentinel the RM_SKIP combine reads
                        // as a
            dest[p] = 0xFFFFFFFFu;  // zero part (no host w x local mask inside the fused program)
#else
            dest[p] = 0;
#endif
        }
    }
#ifdef PLAN_SORTED
    {
        uint32_t o = 0;
        for (uint32_t e = 0; e < EP; ++e) {
            lfill[e] = o;
            o += hist[i * EP + e];
        }
        uint32_t t = p0 % C;
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t e = ids[p] - lo;
            if (e < E) {
#ifdef PMK_OUT_TM
                sorted[lfill[e]] = p0 + p;  // Stage 14 B2.3 token-major out_rm: disp carries the pair (t = p mod C)
#else
                sorted[lfill[e]] = t;
#endif
                lfill[e] += 1;
            }
            t = (t + 1 == C) ? 0 : t + 1;
        }
    }
#endif
    meta[0] = s;
    for (uint32_t w = 1; w < 32; ++w) {
        meta[w] = 0;
    }
    asm volatile("fence" ::: "memory");  // every table is in L1 before the NoC reads it
    if (n > 0) {
        const InterleavedAddrGen<true> dest_g{.bank_base_address = dest_addr, .page_size = N * 4};
        noc_async_write(reinterpret_cast<uint32_t>(dest), dest_g.get_noc_addr(0, p0 * 4), n * 4);
#ifdef PLAN_SORTED
        const InterleavedAddrGen<true> stage_g{.bank_base_address = stage_addr, .page_size = CHUNK * 4};
        noc_async_write(reinterpret_cast<uint32_t>(sorted), stage_g.get_noc_addr(i), n * 4);
#endif
    }
    if (i == 0) {
        const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
        const InterleavedAddrGen<true> meta_g{.bank_base_address = meta_addr, .page_size = 32 * 4};
        noc_async_write(reinterpret_cast<uint32_t>(cnt), cnt_g.get_noc_addr(0), EP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(start), start_g.get_noc_addr(0), EP * 4);
        noc_async_write(reinterpret_cast<uint32_t>(meta), meta_g.get_noc_addr(0), 32 * 4);
    }
    noc_async_write_barrier();
}
