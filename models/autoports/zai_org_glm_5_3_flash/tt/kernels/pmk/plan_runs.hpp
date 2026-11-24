// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK phase form of kernels/prefill/moe_plan_mc_runs.cpp (tt/pmk_moe.py; kernels/pmk/moe_common.hpp): the ORIGINAL
// kernel's arithmetic, CB roles and argument order, with its compile-time args at CT0 + i, its runtime args at RT0 + i
// (the including composite defines both before this header, inside the phase's namespace) and its CBs at pmk_moe's
// indices. Generated from the original by doc/ttnn/16-prefill-production/pmk/moe/gen_phases.py -- edit the original,
// regenerate.
// ---- original header follows ----
// Stage 16 multi-core routing plan, runs variant (GLM53_PF_MOE_PLAN=mc2), program 3 of 3: the dispatch table built
// from the per-chunk expert-sorted token runs that moe_plan_mc_dest.cpp (PLAN_SORTED) staged, instead of every core
// scanning all N pairs (moe_plan_mc_disp.cpp). Core c owns experts [c EPC, min((c + 1) EPC, E)): their regions are
// contiguous in disp ([start[e0], start[e1]) -- regions in expert order), and expert e's rows are the runs
// staging[i][loff_i(e) .. + hist[i][e]) for chunks i = 0 .. P - 1 in order (= pair order), then padding (C). The last
// core also writes the C tail [rows used, CAP). Same table as moe_plan_v2.cpp.
// Per chunk i the owned experts' tokens are ONE contiguous span of staging row i: P span reads, one barrier, then the
// runs are copied into the regions in L1.
// Compile-time: E, EP, CHUNK, P, C, CAP, EPC, RMAX (words of the owned regions' L1 buffer, >= EPC C). Runtime: hist,
// cnt, start, meta, staging, disp addresses, core c.
// Scratch CB CB_PLAN: P EP + 2 EP + 32 + RMAX + (RMAX + 32 P) + 3 P words.
// RMAX < EPC C (GLM53_PMK_PLAN_WIN, Stage 14 W1: program B over a whole-prompt group of rows): the WINDOWED form -- the
// owned regions are assembled RMAX words at a time (disp = the output window, pre-filled with C) from per-expert run
// reads (scratch = RMAX + 32 P words, >= one chunk's run); the same table, with L1 independent of C.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

FORCE_INLINE void phase_main() {
    constexpr uint32_t E = get_compile_time_arg_val(CT0 + 0);
    constexpr uint32_t EP = get_compile_time_arg_val(CT0 + 1);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(CT0 + 2);
    constexpr uint32_t P = get_compile_time_arg_val(CT0 + 3);
    constexpr uint32_t C = get_compile_time_arg_val(CT0 + 4);
    constexpr uint32_t CAP = get_compile_time_arg_val(CT0 + 5);
    constexpr uint32_t EPC = get_compile_time_arg_val(CT0 + 6);
    constexpr uint32_t RMAX = get_compile_time_arg_val(CT0 + 7);
    static_assert(EP % 16 == 0 && CHUNK % 16 == 0 && RMAX % 16 == 0, "64-B aligned tables");
    const uint32_t hist_addr = get_arg_val<uint32_t>(RT0 + 0);
    const uint32_t cnt_addr = get_arg_val<uint32_t>(RT0 + 1);
    const uint32_t start_addr = get_arg_val<uint32_t>(RT0 + 2);
    const uint32_t meta_addr = get_arg_val<uint32_t>(RT0 + 3);
    const uint32_t stage_addr = get_arg_val<uint32_t>(RT0 + 4);
    const uint32_t disp_addr = get_arg_val<uint32_t>(RT0 + 5);
    const uint32_t c = get_arg_val<uint32_t>(RT0 + 6);
    const uint32_t e0 = c * EPC;
    if (e0 >= E) {
        return;  // no experts: nothing read, nothing in flight
    }
    const uint32_t e1 = E - e0 < EPC ? E : e0 + EPC;
    const bool last = e1 == E;
#ifndef PMK_PLAN_RUNS_NB
#define PMK_PLAN_RUNS_NB 1
#endif
    // Stage 14 B2.3 PMK_PLAN_RUNS_NB: the owned experts in NB batches of EPB (RMAX = one batch's worst case: EPB x C),
    // so a plan on few cores fits L1; each batch fills, gathers and writes its own disp range -- the same rows, same
    // order
    constexpr uint32_t EPB = (EPC + PMK_PLAN_RUNS_NB - 1) / PMK_PLAN_RUNS_NB;

    const uint32_t base_l1 = get_write_ptr(CB_PLAN);
    uint32_t* hist = reinterpret_cast<uint32_t*>(base_l1);  // [P][EP]
    uint32_t* cnt = hist + P * EP;
    uint32_t* start = cnt + EP;
    uint32_t* meta = start + EP;               // 32 words
    uint32_t* disp = meta + 32;                // RMAX words: the owned regions, then (last core) C tail chunks
    uint32_t* scratch = disp + RMAX;           // RMAX + 32 P words: the P chunks' spans, each 64-B aligned
    uint32_t* sofs = scratch + RMAX + 32 * P;  // per chunk: span start in scratch (words)
    uint32_t* cur = sofs + P;                  // per chunk: the next staged token's index in scratch

    const InterleavedAddrGen<true> hist_g{.bank_base_address = hist_addr, .page_size = EP * 4};
    for (uint32_t r = 0; r < P; ++r) {
        noc_async_read(hist_g.get_noc_addr(r), reinterpret_cast<uint32_t>(hist + r * EP), EP * 4);
    }
    const InterleavedAddrGen<true> cnt_g{.bank_base_address = cnt_addr, .page_size = EP * 4};
    const InterleavedAddrGen<true> start_g{.bank_base_address = start_addr, .page_size = EP * 4};
    const InterleavedAddrGen<true> meta_g{.bank_base_address = meta_addr, .page_size = 32 * 4};
    noc_async_read(cnt_g.get_noc_addr(0), reinterpret_cast<uint32_t>(cnt), EP * 4);
    noc_async_read(start_g.get_noc_addr(0), reinterpret_cast<uint32_t>(start), EP * 4);
    noc_async_read(meta_g.get_noc_addr(0), reinterpret_cast<uint32_t>(meta), 32 * 4);
    noc_async_read_barrier();
    const uint32_t used = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta);
    const InterleavedAddrGen<true> stage_g{.bank_base_address = stage_addr, .page_size = CHUNK * 4};
    const InterleavedAddrGen<true> disp_g{.bank_base_address = disp_addr, .page_size = CAP * 4};
    // (cf3h-k, 2026-10-26) phase_main is not a template: a static_assert inside a discarded if-constexpr branch is
    // still checked, so the windowed form's two conditions live here, each OR'ed with the direct form's selector (the
    // merged build failed every PMK_PLAN_RUNS_NB > 1 compile -- W7's B2.3 plan_runs_nb 2 -- on the windowed branch's
    // assert)
    static_assert(
        RMAX >= EPB * C || PMK_PLAN_RUNS_NB == 1,
        "the windowed form (RMAX < EPC C, GLM53_PMK_PLAN_WIN) runs one batch");
    static_assert(
        RMAX >= EPB * C || RMAX + 32 * P >= CHUNK + 32,
        "one chunk's run (+ the 64-B read alignment) must fit the windowed read scratch");
    if constexpr (RMAX >= EPB * C) {  // one batch's regions fit the L1 buffer (NB = 1: the original direct form)
        for (uint32_t b0 = e0; b0 < e1; b0 += EPB) {  // PMK_PLAN_RUNS_NB = 1: one batch = the original single pass
            const uint32_t b1 = e1 - b0 < EPB ? e1 : b0 + EPB;
            const uint32_t r0 = start[b0];
            const uint32_t r1 = b1 < EP ? start[b1] : used;
            for (uint32_t d = 0; d < r1 - r0; ++d) {
                disp[d] = C;
            }
            uint32_t so = 0;
            for (uint32_t i = 0; i < P; ++i) {
                uint32_t lo = 0;  // this chunk's tokens of the experts below b0
                for (uint32_t e = 0; e < b0; ++e) {
                    lo += hist[i * EP + e];
                }
                uint32_t hi = lo;
                for (uint32_t e = b0; e < b1; ++e) {
                    hi += hist[i * EP + e];
                }
                const uint32_t w0 = lo & ~15u;
                sofs[i] = so;
                cur[i] = so + (lo - w0);
                if (hi > lo) {
                    const uint32_t span = (hi + 15u) / 16u * 16u - w0;
                    noc_async_read(stage_g.get_noc_addr(i, w0 * 4), reinterpret_cast<uint32_t>(scratch + so), span * 4);
                    so += span;
                }
            }
            noc_async_read_barrier();
            for (uint32_t e = b0; e < b1; ++e) {
                uint32_t off = start[e] - r0;
                for (uint32_t i = 0; i < P; ++i) {
                    const uint32_t run = hist[i * EP + e];
                    const uint32_t* src = scratch + cur[i];
                    for (uint32_t w = 0; w < run; ++w) {
                        disp[off + w] = src[w];
                    }
                    off += run;
                    cur[i] += run;
                }
            }
            asm volatile("fence" ::: "memory");  // the table is in L1 before the NoC reads it
            if (r1 > r0) {
                noc_async_write(reinterpret_cast<uint32_t>(disp), disp_g.get_noc_addr(0, r0 * 4), (r1 - r0) * 4);
            }
            noc_async_write_barrier();  // (the next batch reuses disp / scratch)
        }
    } else {
        const uint32_t r0 = start[e0];
        const uint32_t r1 = e1 < EP ? start[e1] : used;
        // the windowed form: disp = the output window [wb, wb + RMAX) of the owned regions [0, r1 - r0) (C = padding);
        // cur[i] = chunk i's next unread staged token (staging row i, words); sofs[i] = its run's start in scratch
        constexpr uint32_t SCAP = RMAX + 32 * P;
        const uint32_t total = r1 - r0;
        uint32_t wb = 0;
        for (uint32_t d = 0; d < RMAX; ++d) {
            disp[d] = C;
        }
        auto flush = [&]() {  // [wb, min(wb + RMAX, total)) -> disp rows r0 + wb ..; then the next window, all C
            const uint32_t n = total - wb < RMAX ? total - wb : RMAX;
            asm volatile("fence" ::: "memory");
            noc_async_write(reinterpret_cast<uint32_t>(disp), disp_g.get_noc_addr(0, (r0 + wb) * 4), n * 4);
            noc_async_write_barrier();
            wb += RMAX;
            for (uint32_t d = 0; d < RMAX; ++d) {
                disp[d] = C;
            }
        };
        for (uint32_t i = 0; i < P; ++i) {
            uint32_t lo = 0;  // this chunk's tokens of the experts below e0
            for (uint32_t e = 0; e < e0; ++e) {
                lo += hist[i * EP + e];
            }
            cur[i] = lo;
        }
        for (uint32_t e = e0; e < e1; ++e) {
            uint32_t pos = start[e] - r0;
            uint32_t i = 0;
            while (i < P) {  // chunks [i, j): the runs whose 64-B aligned spans fit the scratch, one barrier
                uint32_t so = 0, j = i;
                for (; j < P; ++j) {
                    const uint32_t run = hist[j * EP + e];
                    if (run == 0) {
                        continue;
                    }
                    const uint32_t w0 = cur[j] & ~15u;
                    const uint32_t span = (cur[j] + run + 15u) / 16u * 16u - w0;
                    if (so + span > SCAP) {
                        break;
                    }
                    noc_async_read(stage_g.get_noc_addr(j, w0 * 4), reinterpret_cast<uint32_t>(scratch + so), span * 4);
                    sofs[j] = so + (cur[j] - w0);
                    so += span;
                }
                noc_async_read_barrier();
                for (uint32_t k = i; k < j; ++k) {
                    uint32_t run = hist[k * EP + e];
                    const uint32_t* src = scratch + sofs[k];
                    cur[k] += run;
                    while (run > 0) {
                        while (pos >= wb + RMAX) {
                            flush();
                        }
                        const uint32_t room = wb + RMAX - pos;
                        const uint32_t m = run < room ? run : room;
                        uint32_t* dst = disp + (pos - wb);
                        for (uint32_t w = 0; w < m; ++w) {
                            dst[w] = src[w];
                        }
                        src += m;
                        pos += m;
                        run -= m;
                    }
                }
                i = j;
            }
        }
        while (wb < total) {
            flush();
        }
    }
    if (last && used < CAP) {  // the C tail past the last region, RMAX words at a time
        for (uint32_t d = 0; d < RMAX; ++d) {
            disp[d] = C;
        }
        asm volatile("fence" ::: "memory");
        for (uint32_t d = used; d < CAP; d += RMAX) {
            const uint32_t n = CAP - d < RMAX ? CAP - d : RMAX;
            noc_async_write(reinterpret_cast<uint32_t>(disp), disp_g.get_noc_addr(0, d * 4), n * 4);
        }
        noc_async_write_barrier();
    }
}
