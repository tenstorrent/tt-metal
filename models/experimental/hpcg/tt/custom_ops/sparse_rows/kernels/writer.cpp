// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse-row op, writer (NCRISC): gathers its stencil slots of every segment from the L1 x window (its offset
// pieces prefetched one segment ahead) and emits each output segment one segment behind. SpMV / gather mode: to the
// DRAM output. GS mode: into this core's own window and into every neighbour window that holds a copy of these rows
// (NoC writes); after each phase it writes its progress into its slot on every neighbour and on itself. At the end
// of a GS launch the owned rows go back to DRAM x.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include "gather.h"
#include "common.h"
#define CLK() (*(volatile uint32_t*)RISCV_DEBUG_REG_WALL_CLOCK_L)

struct Pend {
    uint32_t on, b, q0, q1, sp;  // sp: segment of an SpMV pass (RESID launches) -> written to OUT2 in DRAM
};

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t meta_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t row = get_absolute_logical_y() * get_common_arg_val<uint32_t>(2) + get_absolute_logical_x();
    const uint32_t off_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t meta_bytes = get_common_arg_val<uint32_t>(10), ntlv = get_common_arg_val<uint32_t>(11);
    const uint32_t flags = get_common_arg_val<uint32_t>(12);
    const uint32_t nph = get_common_arg_val<uint32_t>(13);  // phases to run, if not 0: a prefix of the table's
    const bool wsplit = flags & F_WSPLIT;
    constexpr auto ao = TensorAccessorArgs<0>();
    constexpr auto am = TensorAccessorArgs<ao.next_compile_time_args_offset()>();
    constexpr auto af = TensorAccessorArgs<am.next_compile_time_args_offset()>();
    const auto OUT = TensorAccessor(ao, out_addr, 4096);
    const auto META = TensorAccessor(am, meta_addr, meta_bytes);  // page `row`: this core's row
    const auto OFF = TensorAccessor(af, off_addr, 1024 * OFFB);
    [[maybe_unused]] constexpr auto adbg0 = TensorAccessorArgs<af.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto avv = TensorAccessorArgs<adbg0.next_compile_time_args_offset()>();
#ifdef OFFW
    const auto V = TensorAccessor(avv, get_common_arg_val<uint32_t>(7), 4096);
#endif
    constexpr auto ao2 = TensorAccessorArgs<avv.next_compile_time_args_offset()>();
    [[maybe_unused]] const auto OUT2 = TensorAccessor(ao2, get_common_arg_val<uint32_t>(8), 4096);  // A x of SpMV passes
    constexpr auto asr = TensorAccessorArgs<ao2.next_compile_time_args_offset()>();
    const auto SRC = TensorAccessor(asr, get_common_arg_val<uint32_t>(9), 4096);  // x, for the odd window regions
#ifdef KDEBUG
    uint32_t T[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t t0 = CLK();
    const uint32_t tstart = t0;
#define TIC() t0 = CLK()
#define TOC(i) T[i] += CLK() - t0
#else
#define TIC()
#define TOC(i)
#endif
    const uint32_t meta = get_write_ptr(CB_META2);
    noc_async_read(META.get_noc_addr(row), meta, meta_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    Meta M;
    M.init(meta);
    if (nph) {
        M.NPH = nph;
    }
    const uint32_t win = get_write_ptr(CB_WIN);
    const uint32_t offw = get_write_ptr(CB_OFF2);  // 2 x count_of(1) offset pages
    constexpr uint32_t C1 = count_of(1);
    Pend pend = {0, 0, 0, 0, 0};
    auto emit = [&](const Pend& e) {
        cb_wait_front(CB_OUT, 1);
        const uint32_t src = get_read_ptr(CB_OUT) + (e.q0 & (TPW - 1)) * 4;
        const uint32_t n = e.q1 - e.q0;
#ifdef GS
        if (e.sp) {  // a segment of an SpMV pass (residual launches): A x goes to OUT2
            noc_async_write(src, OUT2.get_noc_addr(e.q0 >> 10) + (e.q0 & 1023) * 4, n * 4);
            noc_async_writes_flushed();
            cb_pop_front(CB_OUT, 1);
            return;
        }
        const volatile uint32_t* bk = M.blk(e.b);
        noc_async_write(src, get_noc_addr(win + bk[6] + (e.q0 - bk[4]) * 4), n * 4);  // this core's own window
        const volatile uint32_t* ex = M.m + bk[2];
        for (uint32_t i = 0; i < bk[3]; ++i) {
            const uint32_t qs = ex[5 * i + 2], qe = ex[5 * i + 3];
            const uint32_t s = qs > e.q0 ? qs : e.q0, f = qe < e.q1 ? qe : e.q1;
            if (s < f)
                noc_async_write(
                    src + (s - e.q0) * 4,
                    get_noc_addr(ex[5 * i], ex[5 * i + 1], win + ex[5 * i + 4] + (s - qs) * 4),
                    (f - s) * 4);
        }
#else
        noc_async_write(src, OUT.get_noc_addr(e.q0 >> 10) + (e.q0 & 1023) * 4, n * 4);
#endif
        // the output slot may be reused once the data has left this core; completion is only needed before a
        // phase is published (publish() waits for it) and at the end of the launch
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    };
    constexpr uint32_t GSZ = (NSLOT + NGRP - 1) / NGRP;
    constexpr uint32_t MG1 = max_in_group(1, GSZ);  // writer offset pages per unit
    auto grp_lo = [](uint32_t g) { return g * GSZ; };
    auto grp_hi = [](uint32_t g) { return (g + 1) * GSZ < NSLOT ? (g + 1) * GSZ : NSLOT; };
#ifdef OFFW
    // the writer also reads the matrix values of its own slots (CB_V2) and the offsets of the pack (ring + mailbox
    // counters) and unpack (CB_OFF4) gatherers, so both data-movement RISCs carry about half of the DRAM reads
    volatile tt_l1_ptr uint32_t* mb = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_MB);
    const uint32_t ring = get_write_ptr(CB_OFF3);
    uint32_t pk_issued = 0;
    uint32_t pk_at[LOOK + 1] = {};  // pk_issued after each in-flight unit's reads were issued
    auto cnt = [&](uint32_t g, uint32_t o) {
        uint32_t c = 0;
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            c += owner(j) == o;
        }
        return c;
    };
#endif
    uint32_t v2pend = 0, o4pend = 0;  // CB pages reserved for issued units, not pushed yet
    (void)v2pend;
    (void)o4pend;
    auto issue = [&](uint32_t u, uint32_t g, uint32_t q0, uint32_t q1) {
        const uint32_t t = q0 >> 10, ob = (q0 & 1023) * OFFB, onb = (q1 - q0) * OFFB;
        const uint32_t lw = (q0 & (TPW - 1)) * 4, lob = (q0 & (TPW - 1)) * OFFB;  // offsets inside L1 pages
        (void)lw;
#ifdef OFFW
        const uint32_t wb = (q0 & 1023) * 4, nb = (q1 - q0) * 4;
        cb_reserve_back(CB_V2, v2pend + cnt(g, 1));
        const uint32_t v2 = cb_ahead_ptr(CB_V2, v2pend, PGB);
        v2pend += cnt(g, 1);
        cb_reserve_back(CB_OFF4, o4pend + cnt(g, 3));
        const uint32_t o4 = cb_ahead_ptr(CB_OFF4, o4pend, OPGB);
        o4pend += cnt(g, 3);
        uint32_t n2 = 0, n4 = 0;
#endif
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            if (owner(j) == 1) {
                noc_async_read(
                    OFF.get_noc_addr(j * ntlv + t) + ob,
                    offw + (USLOT(u) * MG1 + rank_in(grp_lo(g), j, 1)) * OPGB + lob,
                    onb);
#ifdef OFFW
                noc_async_read(V.get_noc_addr(j * ntlv + t) + wb, v2 + n2 * PGB + lw, nb);
                ++n2;
            } else if (owner(j) == 3) {
                noc_async_read(OFF.get_noc_addr(j * ntlv + t) + ob, o4 + n4 * OPGB + lob, onb);
                ++n4;
            } else if (owner(j) == 2) {
                while (pk_issued >= RING && pk_issued + 1 - mb[3] > RING) invalidate_l1_cache();  // ring starts empty
                noc_async_read(OFF.get_noc_addr(j * ntlv + t) + ob, ring + (pk_issued % RING) * OPGB + lob, onb);
                ++pk_issued;
#endif
            }
        }
    };
    // issue cursor: unit ui = (segment walk, group ig), LOOK units ahead of the unit being gathered
    auto load = [&](uint32_t& pp, uint32_t& kk, uint32_t& nn, const volatile uint32_t*& sg) {
        while (pp < M.NPH) {
            const volatile uint32_t* bk = M.blk(M.blkid(pp));
            sg = M.m + bk[0];
            nn = bk[1];
            kk = 0;
            if (nn) {
                return;
            }
            ++pp;
        }
    };
    auto adv = [&](uint32_t& pp, uint32_t& kk, uint32_t& nn_, const volatile uint32_t*& sg) {
        if (++kk == nn_) {
            ++pp;
            load(pp, kk, nn_, sg);
        }
    };
    uint32_t ip = 0, ik = 0, in_ = 0, ig = 0, ui = 0;
    const volatile uint32_t* iseg = nullptr;
    load(ip, ik, in_, iseg);
    auto issue_next = [&]() {
        if (ip >= M.NPH) {
            return;
        }
        noc_async_read_set_trid(TRID(ui));
        issue(ui, ig, iseg[2 * ik], iseg[2 * ik + 1]);
#ifdef OFFW
        pk_at[USLOT(ui)] = pk_issued;
#endif
        ++ui;
        if (++ig == NGRP) {
            ig = 0;
            adv(ip, ik, in_, iseg);
        }
    };
    bool early = false;
    if (!wsplit) {
        // my first units' reads now (matrix values / offsets do not depend on the window)
        for (uint32_t l = 0; l < LOOK; ++l) {
            issue_next();
        }
        early = true;
    }
    TIC();
    if (wsplit) {
        const uint32_t s0 = get_semaphore(0);
        if (!(flags & F_ZEROX)) {
            const uint32_t wn = get_write_ptr(CB_WIN);
            for (uint32_t r = 1; r < M.R; r += 2) {
                read_q(SRC, M.lo(r), wn + M.l1(r), M.len(r));
            }
            noc_async_read_barrier();
        }
        noc_semaphore_inc(get_noc_addr(s0), 1);
        noc_async_atomic_barrier();
    }
    cb_wait_front(CB_RDY, 1);
    TOC(0);
    invalidate_l1_cache();
    // the same flattened (phase, segment) walk as the reader
    uint32_t p = 0, s = 0, k = 0, n = 0;
    const volatile uint32_t* seg = nullptr;
    load(p, k, n, seg);
    uint32_t done = 0;  // phases whose completion this core has published
    const uint32_t NGS = M.ngs();
    const uint32_t sem0 = get_semaphore(0);
    // publish progress value v to this core's slot 0 and to my slot on every neighbour (after my data writes landed)
    auto publish = [&](uint32_t v) {
        noc_async_write_barrier();
        *(volatile tt_l1_ptr uint32_t*)(sem0 + 16) = v;
        const volatile uint32_t* nbs = M.nb();
        for (uint32_t i = 0; i < M.NNB; ++i) {
            noc_inline_dw_write(get_noc_addr(nbs[3 * i], nbs[3 * i + 1], sem0 + (1 + nbs[3 * i + 2]) * 16), v);
        }
        noc_async_write_barrier();
    };
    auto signal = [&](uint32_t cnt) { publish(done + cnt + POFF); };
#ifdef GS
    publish(1);  // round 1: my window is loaded
#endif
    (void)signal;
    (void)done;
    uint32_t u = 0, g = 0;
    uint32_t published = 0;  // units whose landed reads were handed to the compute side (count)
    (void)published;
    if (!early) {
        for (uint32_t l = 0; l < LOOK; ++l) {
            issue_next();
        }
    }
    while (p < M.NPH) {
        const uint32_t b = M.blkid(p);
        const uint32_t spph = M.sp(p);
        const uint32_t q0 = seg[2 * k], q1 = seg[2 * k + 1];
        const uint32_t w0 = q0 & (TPW - 1), w1 = w0 + (q1 - q0);
#ifdef OFFW
        if (published != u + 1) {  // not already published early (during the previous unit's gather)
            TIC();
            noc_async_read_barrier_with_trid(TRID(u));
            invalidate_l1_cache();
            TOC(4);
            cb_push_back(CB_V2, cnt(g, 1));
            cb_push_back(CB_OFF4, cnt(g, 3));
            v2pend -= cnt(g, 1);
            o4pend -= cnt(g, 3);
            mb[2] = pk_at[USLOT(u)];  // only entries whose reads have landed (units <= u)
            published = u + 1;
        }
#endif
        TIC();
        issue_next();  // unit u + LOOK: matrix / offset reads do not depend on the phase barrier below
        TOC(2);
#ifdef GS
        if (g == 0 && p > done) {
            // phases done..p-1 are finished on this core (some may have had no rows here): last output of the
            // previous phase, then publish the progress to every neighbour and to this core's reader
            if (pend.on) {
                TIC();
                emit(pend);
                TOC(6);
                pend.on = 0;
            }
            signal(p - done);
            done = p;
            TIC();
            wait_slots(sem0, M.NNB + 1, p + POFF);  // SpMV passes too: lockstep (see the reader)
            TOC(1);
        }
#endif
#ifndef OFFW
        TIC();
        noc_async_read_barrier_with_trid(TRID(u));
        invalidate_l1_cache();
        TOC(4);
#endif
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            if (owner(j) != 1) {
                continue;
            }
            TIC();
            cb_reserve_back(CB_G2, 1);
            TOC(5);
            TIC();
            gather_tile(
                win,
                (const uint32_t*)(offw + (USLOT(u) * MG1 + rank_in(grp_lo(g), j, 1)) * OPGB),
                (uint32_t*)get_write_ptr(CB_G2),
                w0,
                w1);
            TOC(3);
            cb_push_back(CB_G2, 1);
#if defined(OFFW)
            // the next unit's reads may have landed already: hand them to the pack / unpack gatherers now
            if (published == u + 1 && u + 1 < ui &&
                ncrisc_noc_read_with_transaction_id_flushed(noc_index, TRID(u + 1))) {
                invalidate_l1_cache();
                const uint32_t gn = (g + 1 == NGRP) ? 0 : g + 1;
                cb_push_back(CB_V2, cnt(gn, 1));
                cb_push_back(CB_OFF4, cnt(gn, 3));
                v2pend -= cnt(gn, 1);
                o4pend -= cnt(gn, 3);
                mb[2] = pk_at[USLOT(u + 1)];
                published = u + 2;
            }
#endif
        }
        ++u;
        if (++g == NGRP) {
            g = 0;
#ifdef GS
            if (s == 0) {
                wait_slots(sem0, M.NNB + 1, POFF);  // every neighbour has loaded (imported) its window
            }
#endif
            TIC();
            if (pend.on) {
                emit(pend);
            }
            TOC(6);
            pend = {1, b, q0, q1, spph};
            adv(p, k, n, seg);
            ++s;
        }
    }
    if (pend.on) {
        emit(pend);
    }
    noc_async_write_barrier();
#ifdef GS
    // the remaining phases are done
    if (M.NPH > done) {
        signal(M.NPH - done);
    }
    // owned rows back to DRAM x once this core's own last-phase writes have landed (none if no GS phases ran)
    noc_async_atomic_barrier();
    for (uint32_t b = 0; b < (NGS ? M.NB : 0u); ++b) {
        const volatile uint32_t* bk = M.blk(b);
        const uint32_t a = bk[4], e = bk[5];
        if (a < e) {
            write_q(OUT, a, win + bk[6], e - a);
        }
    }
    noc_async_write_barrier();
#endif
#ifdef KDEBUG
    T[7] = CLK() - tstart;
    {
        constexpr auto adbg = TensorAccessorArgs<af.next_compile_time_args_offset()>();
        const auto DBG = TensorAccessor(adbg, get_common_arg_val<uint32_t>(6), 64);
        volatile uint32_t* o = (volatile uint32_t*)(get_write_ptr(CB_MB) + 48);
        for (uint32_t i = 0; i < 8; ++i) {
            o[i] = T[i];
        }
        noc_async_write(get_write_ptr(CB_MB) + 48, DBG.get_noc_addr(row * 2 + 1), 32);
        noc_async_write(get_write_ptr(CB_MB) + 80, DBG.get_noc_addr(row * 2 + 1) + 32, 32);  // pack / unpack counters
        noc_async_write_barrier();
    }
#endif
}
