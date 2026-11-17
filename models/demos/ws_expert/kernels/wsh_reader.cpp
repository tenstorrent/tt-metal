// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert shared expert, cluster core NCRISC (NOC1: in-cluster traffic and the posted row writes; no DRAM reads, so
// nothing from outside the cluster rectangle lands on its multicast NoC). A non-blocking loop over:
//   QB x ring: push block s (CB 0) once its feeder's multicast flag of slot s % NX reads s + 1; when compute pops
//      block c, c + 1 goes into this core's word of the XPROG of block c + NX's feeder (the feeder of row tile i is
//      i % NF), which multicasts that block once every cluster core has popped block c;
//   QH h: this core's h tiles of row i (CB 16, NPAIR tiles) into the hub's h ring slot i % NS at this core's tiles,
//      then, once acknowledged, + 1 on the slot's arrival count at the hub. No slot credits: a core's h row i needs
//      all its x blocks of row i, and the hub multicast the last one once every cluster core had popped block
//      NB (i + 1) - NX; with NX <= 2 NB - 1 that is in row i - 1, so every core is in the gate/up of row i - 1 or
//      later and has finished the down of row i - 3 (NS = 3): the hub's multicast of row i - 3 has left;
//   QR h ring: push row q (CB 2, KTD tiles) once the hub's flag of slot q % NS reads q + 1;
//   QY rows out: each row-major row tile of CB 17 (32 rows x DC * 32 columns) to the result rows t S + KSH at column
//      tile C0, posted writes (no acks into the rectangle), tokens t < T only; fenced at the end by an acknowledged
//      4-B write into every DRAM bank (the result buffer's spare rows from T S) on the same NoC and VC.
// RT: 0 J, 1 rows, 2 x blocks of the cluster, 3 C0, 4 cluster CL, 5 result addr, 6 T, 7 NF, 8 + f: feeder f's NoC
//     x | y << 8 (feeder 0: the hub).
// CT: 0 NPAIR, 1 BK, 2 KTD, 3 DC, 4 NS, 5 NX, 6 xflags, 7 xprog, 8 hring, 9 hflags, 10 hcnt, 11 h tile bytes, 12 S,
//     13 KSH, 14 result row bytes, 15 NCL, 16 KT, then TensorAccessorArgs(result).
// WSH_FOLD (the shared expert folded into the end pass, ws_fold.py): QY sends each row tile's rows not to the result
//   buffer but into the L1 ring of the end core that owns the row tile, half a row tile a slot: the cluster's row
//   tile i is end core e = i % NE's k = i / NE-th; its half h (rows 16 h .. 16 h + 15) is the end core's half
//   m = 2 k + h, in slot h, rows of rowb bytes, this core's columns at C0 64; then m + 1 into the slot's flag word for
//   this core (J), both posted through the same command buffer (the flag lands after the rows). Slot h is free for
//   half m once the end core's credit word e in this core's FCRED reads m - 1 (it finished half m - 2). No DRAM
//   writes, no fence. Extra CT (after the accessor): NE, the end ring, its flags, FCRED, a scratch word (the flag's
//   source), GC; extra RT (after the feeders): the cluster's NE end cores' NoC x | y << 8.
// A launch starts with this core's state words zeroed (x and h flags; WSH_FOLD: FCRED - their L1 is another program's
// between launches), then + 1 on the master's semaphore 2 (the first hub; WSH_FOLD: end core 0), and waits for
// semaphore 3 (the master sets it once every member cleared its words). RT: the master's NoC x | y << 8 last.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NPAIR = get_compile_time_arg_val(0);
    constexpr uint32_t BK = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t DC = get_compile_time_arg_val(3);
    constexpr uint32_t NS = get_compile_time_arg_val(4);
    constexpr uint32_t NX = get_compile_time_arg_val(5);
    constexpr uint32_t xflags = get_compile_time_arg_val(6);
    constexpr uint32_t xprog = get_compile_time_arg_val(7);
    constexpr uint32_t hring = get_compile_time_arg_val(8);
    constexpr uint32_t hflags = get_compile_time_arg_val(9);
    constexpr uint32_t hcnt = get_compile_time_arg_val(10);
    constexpr uint32_t htile = get_compile_time_arg_val(11);
    constexpr uint32_t S = get_compile_time_arg_val(12);
    constexpr uint32_t KSH = get_compile_time_arg_val(13);
    constexpr uint32_t rowb = get_compile_time_arg_val(14);
    constexpr uint32_t NCL = get_compile_time_arg_val(15);
    constexpr uint32_t KT = get_compile_time_arg_val(16);
    constexpr auto rargs = TensorAccessorArgs<17>();
#ifdef WSH_FOLD
    constexpr uint32_t fo = rargs.next_compile_time_args_offset();
    constexpr uint32_t NE = get_compile_time_arg_val(fo);
    constexpr uint32_t ering = get_compile_time_arg_val(fo + 1);
    constexpr uint32_t eflags = get_compile_time_arg_val(fo + 2);
    constexpr uint32_t fcred = get_compile_time_arg_val(fo + 3);
    constexpr uint32_t fscr = get_compile_time_arg_val(fo + 4);
    constexpr uint32_t GCW = get_compile_time_arg_val(fo + 5);
#endif
    constexpr uint32_t xr_cb = tt::CBIndex::c_0, hr_cb = tt::CBIndex::c_2;
    constexpr uint32_t h_cb = tt::CBIndex::c_16, y_cb = tt::CBIndex::c_17;
    constexpr uint32_t seg = DC * 64;  // a row's bytes at this core's columns
    constexpr uint32_t H_TRID = 0xE;

    const uint32_t J = get_arg_val<uint32_t>(0), rows = get_arg_val<uint32_t>(1), nblk = get_arg_val<uint32_t>(2);
    const uint32_t C0 = get_arg_val<uint32_t>(3), CL = get_arg_val<uint32_t>(4), T = get_arg_val<uint32_t>(6);
    const auto rs = TensorAccessor(rargs, get_arg_val<uint32_t>(5), rowb);
    const uint32_t NF = get_arg_val<uint32_t>(7);
    const uint32_t hub = get_arg_val<uint32_t>(8), hx = hub & 0xFF, hy = hub >> 8;
    auto word = [](uint32_t base, uint32_t i) {
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + i * 16);
    };

    {  // the start: state words zeroed, then the barrier
        for (uint32_t i = 0; i < NX; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + i * 16) = 0;
        }
        for (uint32_t i = 0; i < NS; ++i) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + i * 16) = 0;
        }
#ifdef WSH_FOLD
        for (uint32_t e = 0; e < NE; ++e) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fcred + e * 16) = 0;
        }
        const uint32_t mxy = get_arg_val<uint32_t>(8 + NF + NE);
#else
        const uint32_t mxy = get_arg_val<uint32_t>(8 + NF);
#endif
        noc_semaphore_inc(get_noc_addr(mxy & 0xFF, mxy >> 8, get_semaphore(2)), 1);
        noc_async_atomic_barrier();
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(3)), 1);
    }
    // ---- QB
    uint32_t b_s = 0, x_credited = 0, x_popped = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    constexpr uint32_t NB = KT / BK;
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < nblk && cb_pages_reservable_at_back(xr_cb, BK)) {
            invalidate_l1_cache();
            if (word(xflags, b_s % NX) >= b_s + 1) {
                cb_reserve_back(xr_cb, BK);
                cb_push_back(xr_cb, BK);
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_popped += (uint16_t)(a - x_last);
        x_last = a;
        while (x_credited < x_popped / BK) {
            const uint32_t nxt = x_credited + NX;
            if (nxt < nblk) {
                const uint32_t fxy = get_arg_val<uint32_t>(8 + (nxt / NB) % NF);
                noc_inline_dw_write(get_noc_addr(fxy & 0xFF, fxy >> 8, xprog + J * 16), x_credited + 1);
            }
            ++x_credited;
            did = true;
        }
        return did;
    };

    // ---- QH
    uint32_t h_i = 0;
    bool h_wait = false;
    auto qh = [&]() -> bool {
        if (h_wait) {
            if (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, H_TRID)) {
                return false;
            }
            noc_semaphore_inc(get_noc_addr(hx, hy, hcnt + (h_i % NS) * 16), 1);
            cb_pop_front(h_cb, NPAIR);
            ++h_i;
            h_wait = false;
            return true;
        }
        if (h_i >= rows || !cb_pages_available_at_front(h_cb, NPAIR)) {
            return false;
        }
        noc_async_write_one_packet_with_trid(
            get_read_ptr(h_cb),
            get_noc_addr(hx, hy, hring + ((h_i % NS) * KTD + J * NPAIR) * htile),
            NPAIR * htile,
            H_TRID);
        noc_async_write_set_trid(0);
        h_wait = true;
        return true;
    };

    // ---- QR
    uint32_t r_q = 0;
    auto qr = [&]() -> bool {
        if (r_q >= rows || !cb_pages_reservable_at_back(hr_cb, KTD)) {
            return false;
        }
        invalidate_l1_cache();
        if (word(hflags, r_q % NS) < r_q + 1) {
            return false;
        }
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
        ++r_q;
        return true;
    };

    // ---- QY
    uint32_t y_i = 0;
    bool y_wait = false;
#ifdef WSH_FOLD
    uint32_t y_h = 0;  // (the half of row tile y_i to send next)
#endif
    auto qy = [&]() -> bool {
        if (y_wait) {
            if (!ncrisc_noc_posted_writes_sent(noc_index)) {
                return false;
            }
            y_wait = false;
#ifdef WSH_FOLD
            if (++y_h < 2) {
                return true;
            }
            y_h = 0;
#endif
            cb_pop_front(y_cb, DC);
            ++y_i;
            return true;
        }
        if (y_i >= rows || !cb_pages_available_at_front(y_cb, DC)) {
            return false;
        }
        const uint32_t r = CL + NCL * y_i;
        uint32_t l1 = get_read_ptr(y_cb);
#ifdef WSH_FOLD
        const uint32_t e = y_i % NE, m = 2 * (y_i / NE) + y_h;
        invalidate_l1_cache();
        if (m >= 2 && word(fcred, e) < m - 1) {
            return false;
        }
        const uint32_t exy = get_arg_val<uint32_t>(8 + NF + e), ex = exy & 0xFF, ey = exy >> 8;
        const uint32_t slot = ering + y_h * 16 * rowb + C0 * 64;
        l1 += y_h * 16 * seg;
        for (uint32_t i = 0; i < 16 && r * 32 + 16 * y_h + i < T; ++i, l1 += seg) {
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(l1, get_noc_addr(ex, ey, slot + i * rowb), seg);
        }
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fscr) = m + 1;
        noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
            fscr, get_noc_addr(ex, ey, eflags + (y_h * GCW + J) * 16), 4);
#else
        for (uint32_t i = 0; i < 32 && r * 32 + i < T; ++i, l1 += seg) {
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                l1, rs.get_noc_addr((r * 32 + i) * S + KSH, C0 * 64), seg);
        }
#endif
        y_wait = true;
        return true;
    };

    while (true) {
        const bool b = qb();
        const bool h = qh();
        const bool r = qr();
        const bool y = qy();
        if (!b && !h && !r && !y && b_s >= nblk && h_i >= rows && r_q >= rows && y_i >= rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_posted_writes_flushed();
#ifndef WSH_FOLD
    // the posted row writes' fence: an acknowledged 4-B write into every DRAM bank (the result buffer's spare rows
    // from T S, nobody reads them) after them on the same NoC and VC; its acks mean every row landed
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(get_semaphore(0), rs.get_noc_addr(T * S + i), 4);
    }
    noc_async_write_barrier();
#endif
}
