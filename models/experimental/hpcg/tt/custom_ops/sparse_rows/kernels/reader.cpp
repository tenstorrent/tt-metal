// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sparse-row op, reader (BRISC). Loads this core's x window pieces once, then walks the phases (GS: one colour each;
// SpMV / gather: all rows) and owned segments (q0, q1). Reads are prefetched one segment ahead: all NSLOT matrix-value
// pieces, the reader's and the pack thread's offset pieces and b are issued together, so DRAM latency is paid once per
// segment and hidden behind the previous segment's gather. Gathers the reader's slots (G[w] = WIN[OFF[w]]). GS mode
// waits, before the first gather of each phase, until every neighbour (and this core's writer) has finished the
// previous phase.
#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include "gather.h"
#include "common.h"
#ifndef RNOC
#define RNOC noc_index
#endif
#define CLK() (*(volatile uint32_t*)RISCV_DEBUG_REG_WALL_CLOCK_L)

struct SegIter {  // flattened (phase, segment) walk over META
    Meta* M;
    uint32_t p, k, n;
    const volatile uint32_t* seg;
    bool first_of_phase;
    void start(Meta* m) {
        M = m;
        p = 0;
        k = 0;
        n = 0;
        load();
    }
    void load() {
        while (p < M->NPH) {
            const volatile uint32_t* bk = M->blk(M->blkid(p));
            seg = M->m + bk[0];
            n = bk[1];
            k = 0;
            first_of_phase = true;
            if (n) {
                return;
            }
            ++p;
        }
    }
    bool done() const { return p >= M->NPH; }
    uint32_t q0() const { return seg[2 * k]; }
    uint32_t q1() const { return seg[2 * k + 1]; }
    void next() {
        first_of_phase = false;
        if (++k == n) {
            ++p;
            load();
        }
    }
};

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t v_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t off_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t meta_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(4);
    // every argument is common to all cores; this core's row of the tables follows from its logical position
    const uint32_t row = get_absolute_logical_y() * get_common_arg_val<uint32_t>(5) + get_absolute_logical_x();
    // grid-dependent sizes come as runtime arguments, so one binary serves every level and grid size
    const uint32_t meta_bytes = get_common_arg_val<uint32_t>(7);  // META's page: one core's row, a multiple of 64 B
    const uint32_t ntlv = get_common_arg_val<uint32_t>(8);
    const uint32_t winb = get_common_arg_val<uint32_t>(9), flags = get_common_arg_val<uint32_t>(10);
    const uint32_t nph = get_common_arg_val<uint32_t>(11);  // phases to run, if not 0: a prefix of the table's
    const bool wsplit = flags & F_WSPLIT, addb = flags & F_ADDB, restf = flags & F_RESTF;
#ifdef GS
    constexpr bool kGS = true;
#else
    constexpr bool kGS = false;
#endif
    constexpr auto as = TensorAccessorArgs<0>();
    constexpr auto av = TensorAccessorArgs<as.next_compile_time_args_offset()>();
    constexpr auto ao = TensorAccessorArgs<av.next_compile_time_args_offset()>();
    constexpr auto am = TensorAccessorArgs<ao.next_compile_time_args_offset()>();
    constexpr auto ab = TensorAccessorArgs<am.next_compile_time_args_offset()>();
    const auto SRC = TensorAccessor(as, src_addr, 4096);  // unused when ZEROX
    const auto V = TensorAccessor(av, v_addr, 4096);
    const auto OFF = TensorAccessor(ao, off_addr, 1024 * OFFB);
    const auto META = TensorAccessor(am, meta_addr, meta_bytes);  // page `row`: this core's row
    const auto Bv = TensorAccessor(ab, b_addr, 4096);
    (void)Bv;
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
    const uint32_t meta = get_write_ptr(CB_META);
    noc_async_read(META.get_noc_addr(row), meta, meta_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    Meta M;
    M.init(meta);
    if (nph) {
        M.NPH = nph;
    }
    const uint32_t win = get_write_ptr(CB_WIN);
    const uint32_t offr = get_write_ptr(CB_OFF);  // 2 x count_of(0) offset pages
    TIC();
    if (flags & F_ZEROX) {  // x = 0 on entry (pre-smoother): zero the windows on chip instead of reading a zero vector from DRAM.
        // 4 KB from the 512-B hardware zero region, then local copies of that 4 KB over the rest
        const uint32_t nbytes = winb;
        const uint64_t zsrc = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < 4096 && o < nbytes; o += MEM_ZEROS_SIZE) {
            noc_async_read(zsrc, win + o, MEM_ZEROS_SIZE);
        }
        noc_async_read_barrier();
        const uint64_t wsrc = get_noc_addr(win);
        for (uint32_t o = 4096; o < nbytes; o += 4096) {
            noc_async_read(wsrc, win + o, (nbytes - o) < 4096 ? nbytes - o : 4096);
        }
    } else {
        // window pieces from DRAM x. WSPLIT: the writer loads the odd pieces on its own NoC, so a load uses both NoCs
        for (uint32_t r = 0; r < M.R; r += wsplit ? 2 : 1) {
            read_q(SRC, M.lo(r), win + M.l1(r), M.len(r), RNOC);
        }
    }
    // without WSPLIT, the window wait moves after the first unit's matrix reads are issued (one combined wait)
    const bool early_on = !wsplit;
    if (wsplit) {
        noc_async_read_barrier(RNOC);
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    TOC(0);
    const uint32_t sem0 = get_semaphore(0);
    volatile tt_l1_ptr uint32_t* mb = (volatile tt_l1_ptr uint32_t*)get_write_ptr(CB_MB);
    const uint32_t ring = get_write_ptr(CB_OFF3);
    mb[2] = 0;
    mb[3] = 0;
#ifdef KDEBUG
    for (uint32_t i = 20; i < 28; ++i) {
        mb[i] = 0;  // pack / unpack thread cycle counters
    }
#endif
    if (wsplit) {
        // semaphore 0 counts the two halves of the window load (reader + writer, NoC atomic increments)
        noc_semaphore_inc(get_noc_addr(sem0), 1);
        noc_async_atomic_barrier();
        wait_sem(sem0, 2);
        cb_reserve_back(CB_RDY, 1);
        cb_push_back(CB_RDY, 1);
    }
    constexpr uint32_t C0 = count_of(0), C2 = count_of(2);
    (void)C0;
    (void)C2;
    (void)ring;

    // Work unit = (segment, slot group g): slots [g*GSZ, (g+1)*GSZ). Its reads (matrix values of the group's slots,
    // offset pieces of the reader's and the pack thread's slots in the group, b with group 0) are issued one unit
    // ahead, so at most ~2 units of reads are in flight per core (bounded NoC demand keeps the slowest cores fed).
    constexpr uint32_t GSZ = (NSLOT + NGRP - 1) / NGRP;
    constexpr uint32_t MG0 = max_in_group(0, GSZ);  // reader offset pages per unit
    uint32_t pk_issued = 0;                         // pack-thread offset pages issued (= ring entries)
    uint32_t pk_at[LOOK + 1] = {};                  // pk_issued after each in-flight unit's reads were issued
    (void)pk_at;
    auto grp_lo = [](uint32_t g) { return g * GSZ; };
    auto grp_hi = [](uint32_t g) { return (g + 1) * GSZ < NSLOT ? (g + 1) * GSZ : NSLOT; };
#ifdef OFFW
    // OFFW: the writer reads the matrix values of its own slots and the offsets of the writer, pack and unpack
    // gatherers, so the two data-movement RISCs issue about half of the DRAM reads each
    auto nv_of = [&](uint32_t g) {
        uint32_t c = 0;
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            c += owner(j) != 1;
        }
        return c;
    };
#else
    auto nv_of = [&](uint32_t g) { return grp_hi(g) - grp_lo(g); };
#endif
#if NG > 3 && !defined(OFFW)
    auto c4_of = [&](uint32_t g) {
        uint32_t c = 0;
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            c += owner(j) == 3;
        }
        return c;
    };
#endif
    uint32_t vpend = 0, bpend = 0, opend = 0;  // CB pages reserved for issued units, not pushed yet
    (void)opend;
    auto issue = [&](uint32_t u, uint32_t g, uint32_t q0, uint32_t q1, bool gsph) {
        (void)gsph;
        const uint32_t t = q0 >> 10, wb = (q0 & 1023) * 4, nb = (q1 - q0) * 4;
        const uint32_t ob = (q0 & 1023) * OFFB, onb = (q1 - q0) * OFFB;
        const uint32_t lw = (q0 & (TPW - 1)) * 4, lob = (q0 & (TPW - 1)) * OFFB;  // offsets inside L1 pages
#if NG > 3 && !defined(OFFW)
        uint32_t n4 = 0;
        cb_reserve_back(CB_OFF4, opend + c4_of(g));
        const uint32_t o4 = cb_ahead_ptr(CB_OFF4, opend, OPGB);
        opend += c4_of(g);
#endif
        const uint32_t lo = grp_lo(g), hi = grp_hi(g);
        cb_reserve_back(CB_V, vpend + nv_of(g));
        const uint32_t vb = cb_ahead_ptr(CB_V, vpend, PGB);
        vpend += nv_of(g);
        uint32_t nv = 0;
        for (uint32_t j = lo; j < hi; ++j) {
#ifdef OFFW
            if (owner(j) == 1)
                continue;
#endif
            noc_async_read(V.get_noc_addr(j * ntlv + t, 0, RNOC) + wb, vb + nv * PGB + lw, nb, RNOC);
            ++nv;
            if (owner(j) == 0)
                noc_async_read(
                    OFF.get_noc_addr(j * ntlv + t, 0, RNOC) + ob,
                    offr + (USLOT(u) * MG0 + rank_in(lo, j, 0)) * OPGB + lob,
                    onb,
                    RNOC);
#if NG > 3 && !defined(OFFW)
            else if (owner(j) == 3) {
                noc_async_read(OFF.get_noc_addr(j * ntlv + t, 0, RNOC) + ob, o4 + n4 * OPGB + lob, onb, RNOC);
                ++n4;
            }
#endif
#if NG > 2 && !defined(OFFW)
            else if (owner(j) == 2) {
                while (pk_issued + 1 - mb[3] > RING) invalidate_l1_cache();
                noc_async_read(
                    OFF.get_noc_addr(j * ntlv + t, 0, RNOC) + ob, ring + (pk_issued % RING) * OPGB + lob, onb, RNOC);
                ++pk_issued;
            }
#endif
        }
        if ((kGS || addb) && g == 0 && gsph) {
            cb_reserve_back(CB_B, bpend + 1);
            noc_async_read(Bv.get_noc_addr(t) + wb, cb_ahead_ptr(CB_B, bpend, PGB) + lw, nb);
            ++bpend;
        }
    };

    SegIter it;
    it.start(&M);
    uint32_t u = 0;  // work unit counter
    uint32_t g = 0;
    SegIter is = it;  // issue cursor: unit ui = (is, ig), LOOK units ahead of the unit being gathered
    uint32_t ig = 0, ui = 0;
    // b is read for Gauss-Seidel units, (RESTF) for the SpMV pass over block 0, whose output is r - A z, and (ADDB)
    // for every output, b + A x (prolongation)
    auto needb = [&](uint32_t p) { return addb || !M.sp(p) || (restf && M.phase(p) == 8); };
    auto issue_next = [&]() {
        if (is.done()) {
            return;
        }
        noc_async_read_set_trid(TRID(ui), RNOC);
        issue(ui, ig, is.q0(), is.q1(), needb(is.p));
        pk_at[USLOT(ui)] = pk_issued;
        ++ui;
        if (++ig == NGRP) {
            ig = 0;
            is.next();
        }
    };
    for (uint32_t l = 0; l < LOOK; ++l) {
        issue_next();
    }
    if (early_on) {
        TIC();
        noc_async_read_barrier(RNOC);  // window + unit 0
        noc_async_read_barrier();
        invalidate_l1_cache();
        TOC(0);
        *(volatile tt_l1_ptr uint32_t*)get_semaphore(0) = 1;  // windows + mailbox ready
        cb_reserve_back(CB_RDY, 1);
        cb_push_back(CB_RDY, 1);
    }
    while (!it.done()) {
        const uint32_t q0 = it.q0(), q1 = it.q1();
        const uint32_t w0 = q0 & (TPW - 1), w1 = w0 + (q1 - q0);
        TIC();
        noc_async_read_barrier_with_trid(TRID(u), RNOC);  // unit u has landed
        invalidate_l1_cache();
        TOC(4);
        cb_push_back(CB_V, nv_of(g));
        vpend -= nv_of(g);
#if NG > 3 && !defined(OFFW)
        cb_push_back(CB_OFF4, c4_of(g));
        opend -= c4_of(g);
#endif
        if ((kGS || addb) && g == 0 && needb(it.p)) {
            cb_push_back(CB_B, 1);
            --bpend;
        }
#if NG > 2 && !defined(OFFW)
        mb[2] = pk_at[USLOT(u)];  // only entries whose reads have landed (units <= u)
#endif
        // next unit: next group of this segment, or group 0 of the next segment
        TIC();
        issue_next();  // unit u + LOOK
        TOC(6);
#ifdef GS
        if (g == 0 && it.first_of_phase && it.p) {
            TIC();
            // neighbours finished phase p - 1; SpMV passes wait too, so all cores stream the matrix in lockstep
            // (free-running SpMV passes streamed DRAM slower: about 430 against 470 GB/s)
            wait_slots(sem0, M.NNB + 1, it.p + POFF);
            TOC(1);
        }
#endif
        for (uint32_t j = grp_lo(g); j < grp_hi(g); ++j) {
            if (owner(j) != 0) {
                continue;
            }
            TIC();
            cb_reserve_back(CB_G, 1);
            TOC(5);
            TIC();
            gather_tile(
                win,
                (const uint32_t*)(offr + (USLOT(u) * MG0 + rank_in(grp_lo(g), j, 0)) * OPGB),
                (uint32_t*)get_write_ptr(CB_G),
                w0,
                w1);
            TOC(3);
            cb_push_back(CB_G, 1);
        }
        ++u;
        if (++g == NGRP) {
            g = 0;
            it.next();
        }
    }
#ifdef KDEBUG
    T[7] = CLK() - tstart;
    {
        constexpr auto adbg = TensorAccessorArgs<ab.next_compile_time_args_offset()>();
        const auto DBG = TensorAccessor(adbg, get_common_arg_val<uint32_t>(6), 64);
        volatile uint32_t* o = (volatile uint32_t*)(get_write_ptr(CB_MB) + 16);
        for (uint32_t k = 0; k < 8; ++k) {
            o[k] = T[k];
        }
        noc_async_write(get_write_ptr(CB_MB) + 16, DBG.get_noc_addr(row * 2), 32);
        noc_async_write_barrier();
    }
#endif
}
