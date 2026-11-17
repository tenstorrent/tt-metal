// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert shared expert, a cluster's x feeders (the rectangle's NF cores without weights; feeder 0, the hub, sits at
// NOC1's multicast start corner and also multicasts the h rows). Feeder f takes the cluster's row tiles i = f, f + NF,
// .. (cluster-local; row tile r = CL + NCL i). Built three times (one source per RISC, WSH_ROLE):
//   WSH_ROLE 0, BRISC (NOC0, DRAM reads): a row tile's 32 token rows (whole rows, H * 2 B each) into CB 7;
//   WSH_ROLE 1, compute: tilizes the row tile (KT tiles wide, CB 7 -> CB 8, x's format, fp32 DEST);
//   WSH_ROLE 2, NCRISC (NOC1): a non-blocking loop over (each multicast into every NOC1 sub-rectangle of the cluster's
//   rectangle: the hub's own with loopback, then the others; payload, then its flag as a 4-B multicast write through
//   the same command buffer, which lands after the payload)
//     QA x: the row tile's NB blocks (s = NB i + b: BK tiles of CB 8) into slot s % NX of every rectangle core's x
//        ring + flag (s + 1), linked; block s waits until every cluster core has popped block s - NX (its word in
//        this feeder's XPROG: cores credit the feeder of the block that reuses the slot), and until block s - 2 has
//        landed here: the feeders' x multicasts go out in stream order, at most two in flight (several linked
//        multicasts in flight into one rectangle starve single calls);
//     QM h (the hub only): row q once every cluster core's h tiles of it landed in this core's h ring slot q % NS
//        (the slot's arrival count, + GC a use) -> the whole row (KTD tiles) into slot q % NS of every rectangle core
//        + flag (q + 1).
// Stamps (each role its words, wall-clock cycles): role 0: 0 start, 1 read waits, 2 CB 7 space waits, 3 end; role 1:
// 8 start, 9 input waits, 10 output space waits, 11 end; role 2: 16 start, 17 x multicasts (issue to sent), 18 h
// multicasts, 19 x blocks waiting for credits, 20 h rows waiting for arrivals, 21 end.
// RT (role 0): 0 x addr, 1 cluster CL, 2 f, 3 NF, 4 cluster rows, 8 stamps; (role 1): 0 own rows, 8 stamps;
//    (role 2): 0 f, 1 NF, 2 cluster rows, 3-6 the hub's NOC1 multicast sub-rectangle, 7 its cores, 8 stamps, 9 hub,
//    10 sub-rectangles, 11 + 5 (i - 1) .. : sub-rectangle i's NOC1 corners and cores; then the start barrier's master
//    (NoC x | y << 8), whether this hub is it, and (the master) the member count and every member's NoC x | y << 8.
// Role 2 starts with the hub's state words zeroed (x and h flags, XPROG, h counts: their L1 is another program's
// between launches), then + 1 on the master's semaphore 2; the master waits for every member, then sets semaphore 3
// on each; all wait for it.
// CT (role 0): 0 KT, 1 NCL, 2 x row bytes, then TensorAccessorArgs(x); (role 1): 0 KT; (role 2): 0 BK, 1 KTD, 2 NX,
//    3 NS, 4 GC, 5 xring, 6 xflags, 7 xprog, 8 hring, 9 hflags, 10 hcnt, 11 x tile bytes, 12 h tile bytes, 13 KT.
#include <cstdint>
#if WSH_ROLE == 1
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#else
#include "api/dataflow/dataflow_api.h"
#endif

constexpr uint32_t xrm_cb = tt::CBIndex::c_7, xst_cb = tt::CBIndex::c_8;

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

#if WSH_ROLE == 0
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t NCL = get_compile_time_arg_val(1);
    constexpr uint32_t xrow = get_compile_time_arg_val(2);
    constexpr auto xargs = TensorAccessorArgs<3>();
    const auto xs = TensorAccessor(xargs, get_arg_val<uint32_t>(0), xrow);
    const uint32_t CL = get_arg_val<uint32_t>(1), f = get_arg_val<uint32_t>(2), NF = get_arg_val<uint32_t>(3);
    const uint32_t rows = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));
    st[0] = wall();
    uint32_t t_rd = 0, t_sp = 0;
    for (uint32_t i = f; i < rows; i += NF) {
        const uint32_t r = CL + NCL * i;
        uint32_t a = wall();
        cb_reserve_back(xrm_cb, KT);
        t_sp += wall() - a;
        uint32_t l1 = get_write_ptr(xrm_cb);
        for (uint32_t j = 0; j < 32; ++j, l1 += xrow) {
            noc_async_read_page(r * 32 + j, xs, l1);
        }
        a = wall();
        noc_async_read_barrier();
        t_rd += wall() - a;
        cb_push_back(xrm_cb, KT);
    }
    st[1] = t_rd;
    st[2] = t_sp;
    st[3] = wall();
}
#elif WSH_ROLE == 1
void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    const uint32_t n = get_arg_val<uint32_t>(0);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));
    UNPACK(st[8] = wall());
    compute_kernel_hw_startup(xrm_cb, xst_cb);
    DataflowBuffer xrm_dfb(xrm_cb), xst_dfb(xst_cb);
    uint32_t t_in = 0, t_out = 0;
    for (uint32_t b = 0; b < n; ++b) {
        uint32_t a = wall();
        cb_wait_front(xrm_cb, KT);
        t_in += wall() - a;
        a = wall();
        cb_reserve_back(xst_cb, KT);
        t_out += wall() - a;
        compute_kernel_lib::tilize<KT, xrm_cb, xst_cb>(xrm_dfb, xst_dfb, 1);
    }
    UNPACK(st[9] = t_in);
    PACK(st[10] = t_out);
    PACK(st[11] = wall());
}
#else
void kernel_main() {
    constexpr uint32_t BK = get_compile_time_arg_val(0);
    constexpr uint32_t KTD = get_compile_time_arg_val(1);
    constexpr uint32_t NX = get_compile_time_arg_val(2);
    constexpr uint32_t NS = get_compile_time_arg_val(3);
    constexpr uint32_t GC = get_compile_time_arg_val(4);
    constexpr uint32_t xring = get_compile_time_arg_val(5);
    constexpr uint32_t xflags = get_compile_time_arg_val(6);
    constexpr uint32_t xprog = get_compile_time_arg_val(7);
    constexpr uint32_t hring = get_compile_time_arg_val(8);
    constexpr uint32_t hflags = get_compile_time_arg_val(9);
    constexpr uint32_t hcnt = get_compile_time_arg_val(10);
    constexpr uint32_t xtile = get_compile_time_arg_val(11);
    constexpr uint32_t htile = get_compile_time_arg_val(12);
    constexpr uint32_t KT = get_compile_time_arg_val(13);
    constexpr uint32_t NB = KT / BK;
    const uint32_t f = get_arg_val<uint32_t>(0), NF = get_arg_val<uint32_t>(1), rows = get_arg_val<uint32_t>(2);
    const uint32_t mx0 = get_arg_val<uint32_t>(3), my0 = get_arg_val<uint32_t>(4);
    const uint32_t mx1 = get_arg_val<uint32_t>(5), my1 = get_arg_val<uint32_t>(6);
    const uint32_t nrec = get_arg_val<uint32_t>(7);
    const bool hub = get_arg_val<uint32_t>(9) != 0;
    const uint32_t nsub = get_arg_val<uint32_t>(10);
    // the payload and its flag into every sub-rectangle (src_flag holds the flag's value)
    auto mcast = [&](uint32_t src, uint32_t dst, uint32_t bytes, uint32_t src_flag, uint32_t dst_flag) {
        noc_async_write_multicast_loopback_src(src, get_noc_multicast_addr(mx0, my0, mx1, my1, dst), bytes, nrec, true);
        noc_async_write_multicast_loopback_src(
            src_flag, get_noc_multicast_addr(mx0, my0, mx1, my1, dst_flag), 4, nrec, false);
        for (uint32_t i = 1; i < nsub; ++i) {
            const uint32_t a = 11 + 5 * (i - 1);
            const uint32_t x0 = get_arg_val<uint32_t>(a), y0 = get_arg_val<uint32_t>(a + 1);
            const uint32_t x1 = get_arg_val<uint32_t>(a + 2), y1 = get_arg_val<uint32_t>(a + 3);
            const uint32_t n = get_arg_val<uint32_t>(a + 4);
            noc_async_write_multicast(src, get_noc_multicast_addr(x0, y0, x1, y1, dst), bytes, n, true);
            noc_async_write_multicast(src_flag, get_noc_multicast_addr(x0, y0, x1, y1, dst_flag), 4, n, false);
        }
    };
    {  // the start: state words zeroed, then the barrier
        auto zero = [](uint32_t base, uint32_t n) {
            for (uint32_t i = 0; i < n; ++i) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * 16) = 0;
            }
        };
        zero(xflags, NX);
        zero(hflags, NS);
        zero(xprog, GC);
        zero(hcnt, NS);
        const uint32_t b0 = 11 + 5 * (nsub - 1);
        const uint32_t mxy = get_arg_val<uint32_t>(b0), cnt = get_semaphore(2), go = get_semaphore(3);
        noc_semaphore_inc(get_noc_addr(mxy & 0xFF, mxy >> 8, cnt), 1);
        noc_async_atomic_barrier();
        if (get_arg_val<uint32_t>(b0 + 1)) {
            const uint32_t n = get_arg_val<uint32_t>(b0 + 2);
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cnt), n);
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t v = get_arg_val<uint32_t>(b0 + 3 + j);
                noc_inline_dw_write(get_noc_addr(v & 0xFF, v >> 8, go), 1);
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(go), 1);
    }
    const uint32_t xscr_addr = get_semaphore(0), hscr_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* xscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xscr_addr);
    volatile tt_l1_ptr uint32_t* hscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hscr_addr);
    auto word = [](uint32_t base, uint32_t i) {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * 16);
    };
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(8));
    st[16] = wall();
    uint32_t t_xm = 0, t_hm = 0, t_cr = 0, t_ar = 0, last = wall();
    bool cr_wait = false, ar_wait = false;
    uint32_t i = f, b = 0, q = hub ? 0 : rows;
    while (i < rows || q < rows) {
        const uint32_t now = wall();
        t_cr += cr_wait ? now - last : 0;
        t_ar += ar_wait ? now - last : 0;
        last = now;
        cr_wait = ar_wait = false;
        if (i < rows && cb_pages_available_at_front(xst_cb, KT)) {
            const uint32_t s = i * NB + b;
            invalidate_l1_cache();
            bool ok = s < 2 || word(xflags, (s - 2) % NX) >= s - 1;
            if (ok && s >= NX) {
                invalidate_l1_cache();
                for (uint32_t c = 0; c < GC && ok; ++c) {
                    ok = word(xprog, c) >= s - NX + 1;
                }
            }
            cr_wait = !ok;
            if (ok) {
                const uint32_t a = wall();
                const uint32_t xs = s % NX;
                *xscr = s + 1;
                mcast(get_read_ptr(xst_cb) + b * BK * xtile, xring + xs * BK * xtile, BK * xtile, xscr_addr, xflags + xs * 16);
                noc_async_writes_flushed();
                if (++b == NB) {
                    cb_pop_front(xst_cb, KT);
                    b = 0;
                    i += NF;
                }
                t_xm += wall() - a;
            }
        }
        if (q < rows) {
            const uint32_t hs = q % NS;
            invalidate_l1_cache();
            ar_wait = word(hcnt, hs) < GC * (q / NS + 1);
            if (!ar_wait) {
                const uint32_t a = wall();
                *hscr = q + 1;
                const uint32_t slot = hring + hs * KTD * htile;
                mcast(slot, slot, KTD * htile, hscr_addr, hflags + hs * 16);
                noc_async_writes_flushed();
                ++q;
                t_hm += wall() - a;
            }
        }
    }
    noc_async_write_barrier();
    st[17] = t_xm;
    st[18] = t_hm;
    st[19] = t_cr;
    st[20] = t_ar;
    st[21] = wall();
}
#endif
