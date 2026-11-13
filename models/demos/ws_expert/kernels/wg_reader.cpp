// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert groups NCRISC (NOC1), a non-blocking loop over:
//   QW weights: when the injector's slice of the group's next expert has landed (WARR >= n + 1: it writes the 2-slot
//      rings directly, then increments WARR - the fabric's fused write + atomic inc), push it to compute (CB 1 gate/up,
//      CB 3 down); once compute popped both parts of an expert, write the freed-expert count into this core's credit
//      word at its injector.
//   QA x multicast: this core's tilized x blocks (CB 8) -> slot s % NX of every group core's x ring (linked payload +
//      flag), once every consumer's popped count (XPROG word) reaches s - NX + 1.
//   QB x ring: push block s once its flag reads >= s + 1; when compute pops block c, write c + 1 into this core's
//      XPROG word at the producer of block c + NX.
//   QH h ring: push row q once both column leaders' flags read >= q + 1.
// RT: 0 j, 1 g, 2 nexp_g, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 injector x, 7 injector y,
//     8 credit word offset (bytes) at the injector
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 GC, 5 G, 6 NS, 7 NX, 8 hflags, 9 xring, 10 xflags, 11 xprog, 12-15 group rect
//     for NOC1 (start = the NOC0 end), 16 x tile bytes, 17 counts L1, 18 injector credits addr
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t SEM_WARR = 0, SEM_SCR2 = 3;

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t BK = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t GC = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t NS = get_compile_time_arg_val(6);
    constexpr uint32_t NX = get_compile_time_arg_val(7);
    constexpr uint32_t hflags = get_compile_time_arg_val(8);
    constexpr uint32_t xring = get_compile_time_arg_val(9);
    constexpr uint32_t xflags = get_compile_time_arg_val(10);
    constexpr uint32_t xprog = get_compile_time_arg_val(11);
    constexpr uint32_t mx0 = get_compile_time_arg_val(12), my0 = get_compile_time_arg_val(13);
    constexpr uint32_t mx1 = get_compile_time_arg_val(14), my1 = get_compile_time_arg_val(15);
    constexpr uint32_t xtile = get_compile_time_arg_val(16);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(17);
    constexpr uint32_t icred = get_compile_time_arg_val(18);
    constexpr uint32_t BPR = KT / BK, HALF = GC / 2;
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS;

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint32_t ix = get_arg_val<uint32_t>(6), iy = get_arg_val<uint32_t>(7), ioff = get_arg_val<uint32_t>(8);

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    auto rows_of = [&](uint32_t n) { return (counts[16 + g + n * G] + 31) / 32; };
    uint32_t total_rows = 0, total_x = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }
    total_x = total_rows * BPR;

    // ---- QW
    uint32_t w_n = 0, w_freed_sent = 0, gu_pages = 0, dn_pages = 0;
    volatile tt_l1_ptr uint32_t* warr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_WARR));
    const uintptr_t gu_ack = (uintptr_t)get_cb_tiles_acked_ptr(w_cb), dn_ack = (uintptr_t)get_cb_tiles_acked_ptr(wd_cb);
    uint16_t gu_last = (uint16_t)reg_read(gu_ack), dn_last = (uint16_t)reg_read(dn_ack);
    const uint64_t cred_dst = get_noc_addr(ix, iy, icred + ioff);
    auto qw = [&]() -> bool {
        bool did = false;
        invalidate_l1_cache();
        if (w_n < nexp_g && *warr >= w_n + 1 && cb_pages_reservable_at_back(w_cb, WGU) &&
            cb_pages_reservable_at_back(wd_cb, WDN)) {
            cb_reserve_back(w_cb, WGU);
            cb_push_back(w_cb, WGU);
            cb_reserve_back(wd_cb, WDN);
            cb_push_back(wd_cb, WDN);
            ++w_n;
            did = true;
        }
        const uint16_t a = (uint16_t)reg_read(gu_ack), b = (uint16_t)reg_read(dn_ack);
        gu_pages += (uint16_t)(a - gu_last);
        dn_pages += (uint16_t)(b - dn_last);
        gu_last = a;
        dn_last = b;
        uint32_t freed = gu_pages / WGU;
        if (dn_pages / WDN < freed) {
            freed = dn_pages / WDN;
        }
        if (freed > w_freed_sent) {
            noc_inline_dw_write(cred_dst, freed);
            w_freed_sent = freed;
            did = true;
        }
        return did;
    };

    // x stream helpers: block s -> (expert, index in it); walked forward only
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_cred = 0, t_empty = 0, n_cred = 0, last_t = wall();
    // ---- QA: own x multicasts
    uint32_t a_n = 0, a_l = J, a_base = 0;
    const uint32_t scr_addr = get_semaphore(SEM_SCR2);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    auto qa = [&]() -> bool {
        while (a_n < nexp_g && a_l >= rows_of(a_n) * BPR) {
            a_base += rows_of(a_n) * BPR;
            ++a_n;
            a_l = J;
        }
        const uint32_t now = wall();
        const uint32_t dt = now - last_t;
        last_t = now;
        if (a_n >= nexp_g) {
            return false;
        }
        if (!cb_pages_available_at_front(xst_cb, BK)) {
            t_empty += dt;
            return false;
        }
        const uint32_t s = a_base + a_l;
        if (s >= NX) {
            invalidate_l1_cache();
            for (uint32_t c = 0; c < GC; ++c) {
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + c * 16) < s - NX + 1) {
                    t_cred += dt;
                    n_cred = c;
                    return false;
                }
            }
        }
        const uint32_t xs = s % NX;
        *scr = s + 1;
        noc_async_write_multicast_loopback_src(
            get_read_ptr(xst_cb), get_noc_multicast_addr(mx0, my0, mx1, my1, xring + xs * BK * xtile), BK * xtile,
            GC, true /*linked*/);
        noc_semaphore_set_multicast_loopback_src(
            scr_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + xs * 16), GC);
        // (flushed, not barrier: only the source must be free before reuse - the flag rides the linked payload; a
        // barrier waited for 20 acks and stalled every other queue of this RISC)
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, BK);
#ifdef WG_TRACE0
        if (s >= WG_TRACE0 && s < WG_TRACE0 + 40) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9))[128 + s - WG_TRACE0] = wall();
        }
#endif
        a_l += GC;
        return true;
    };
    // ---- QB: x ring pushes + credits
    uint32_t b_s = 0, x_popped_pages = 0, x_credited = 0, w_e = 0, w_base = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_x && cb_pages_reservable_at_back(xr_cb, BK)) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + (b_s % NX) * 16) >= b_s + 1) {
                cb_reserve_back(xr_cb, BK);
                cb_push_back(xr_cb, BK);
#ifdef WG_TRACE0
                if (b_s >= WG_TRACE0 && b_s < WG_TRACE0 + 40) {
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9))[192 + b_s - WG_TRACE0] =
                        wall();
                }
#endif
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_popped_pages += (uint16_t)(a - x_last);
        x_last = a;
        while (x_credited < x_popped_pages / BK) {
            const uint32_t nxt = x_credited + NX;
            if (nxt < total_x) {
                while (nxt >= w_base + rows_of(w_e) * BPR) {
                    w_base += rows_of(w_e) * BPR;
                    ++w_e;
                }
                const uint32_t owner = (nxt - w_base) % GC;
                noc_inline_dw_write(
                    get_noc_addr(cxs[owner / HALF], ny0 + owner % HALF, xprog + J * 16), x_credited + 1);
            }
            ++x_credited;
            did = true;
        }
        return did;
    };
    // ---- QH: h ring rows
    uint32_t h_q = 0;
    auto qh = [&]() -> bool {
        if (h_q >= total_rows || !cb_pages_reservable_at_back(hr_cb, KTD)) {
            return false;
        }
        invalidate_l1_cache();
        for (uint32_t c = 0; c < 2; ++c) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + ((h_q % NS) * 2 + c) * 16) < h_q + 1) {
                return false;
            }
        }
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
        ++h_q;
        return true;
    };
    while (true) {
        const bool a = qw();
        const bool b = qa();
        const bool c = qb();
        const bool d = qh();
        if (!a && !b && !c && !d && w_n >= nexp_g && w_freed_sent >= nexp_g && a_n >= nexp_g && b_s >= total_x &&
            x_credited >= total_x && h_q >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(9));
    st[48] = t_cred;
    st[49] = t_empty;
}
