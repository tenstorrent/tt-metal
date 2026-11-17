// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert head (ws_router.py), BRISC (NOC0: DRAM reads, and the x pieces up the column into the hub). Core c of a
// router group of GS cores:
//   - its router weights: W_r^T's expert tiles [EPC c, EPC (c + 1)), all KT K tiles each (CB 0, K-major: expert tile
//     e's K tile k at k EPC + e), pushed a piece of BK K tiles at a time between the x pieces (the first row tile's
//     router starts on the first piece), and those experts' id tiles (CB 6);
//   - x: a row tile's K tiles in NP = KT / BK pieces; piece p of each of the group's row tiles is core p % GS's. For
//     each of mine, in stream order (s = NP i + p): its 32 row pieces of BK 64 B (DRAM) into CB 3 for the compute's
//     tilize; the tilized piece (CB 2) into the hub's relay slot s % NR once the slot is free (s < NR, or my RGRANT
//     word from the hub reads s + 1), then s + 1 into the slot's flag word at the hub. Both posted (no acks beside
//     the DRAM reads on this NoC) and through the same command buffer, on one NoC and virtual channel: the flag lands
//     after its payload. The hub (the group's NOC1 multicast start end) multicasts the piece into every group core's
//     x ring. NOC0 runs toward the hub's end of the column: no wrap. The first relay waits for the start barrier (the
//     NCRISC's: semaphore 6), every core's relay flags cleared.
// RT: 0 x, 1 W_r^T, 2 ids addrs, 3 the group's first row tile, 4 its row tiles, 5 c, 6 stamps (0: start, 5: weights
//     in; wall-clock cycles: 6 waiting for x reads, 7 waiting for relay slots), 7 the hub's NoC x | y << 8.
// CT: 0 KT, 1 BK, 2 EPC, 3 GS, 4 x row bytes, 5 NR, 6 relay ring, 7 relay flags, 8 RGRANT, 9 a scratch word (the
//     flag's source), then TensorAccessorArgs(x), (W_r^T), (ids).
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE uint32_t wall() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); }

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t BK = get_compile_time_arg_val(1);
    constexpr uint32_t EPC = get_compile_time_arg_val(2);
    constexpr uint32_t GS = get_compile_time_arg_val(3);
    constexpr uint32_t xrow = get_compile_time_arg_val(4);
    constexpr uint32_t NR = get_compile_time_arg_val(5);
    constexpr uint32_t relay = get_compile_time_arg_val(6);
    constexpr uint32_t rflags = get_compile_time_arg_val(7);
    constexpr uint32_t rgrant = get_compile_time_arg_val(8);
    constexpr uint32_t bscr = get_compile_time_arg_val(9);
    constexpr auto xargs = TensorAccessorArgs<10>();
    constexpr auto wargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t w_cb = tt::CBIndex::c_0, tz_cb = tt::CBIndex::c_2, xs_cb = tt::CBIndex::c_3;
    constexpr uint32_t id_cb = tt::CBIndex::c_6;
    constexpr uint32_t NP = KT / BK, piece = BK * 64, tile = 2048;
    static_assert(KT % BK == 0);

    const auto xs = TensorAccessor(xargs, get_arg_val<uint32_t>(0), xrow);
    const auto wt = TensorAccessor(wargs, get_arg_val<uint32_t>(1), tile);
    const auto it = TensorAccessor(iargs, get_arg_val<uint32_t>(2), tile);
    const uint32_t R0 = get_arg_val<uint32_t>(3), n = get_arg_val<uint32_t>(4), c = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* stamps = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    const uint32_t hub = get_arg_val<uint32_t>(7), hx = hub & 0xFF, hy = hub >> 8;
    volatile tt_l1_ptr uint32_t* grant = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rgrant);
    stamps[0] = wall();

    cb_reserve_back(id_cb, EPC);
    for (uint32_t e = 0; e < EPC; ++e) {
        noc_async_read_page(c * EPC + e, it, get_write_ptr(id_cb) + e * tile);
    }
    noc_async_read_barrier();
    cb_push_back(id_cb, EPC);
    cb_reserve_back(w_cb, EPC * KT);
    const uint32_t wl1 = get_write_ptr(w_cb);
    uint32_t w_p = 0;  // (W_r^T pieces pushed)

    // my pieces in stream order: q -> row tile q / PPC, piece c + GS (q % PPC)
    constexpr uint32_t PPC_MAX = (NP + GS - 1) / GS;
    const uint32_t PPC = c < NP ? (NP - c + GS - 1) / GS : 0;
    static_assert(PPC_MAX >= 1);
    const uint32_t total = n * PPC;
    auto stream = [&](uint32_t q) { return (q / PPC) * NP + c + GS * (q % PPC); };
    uint32_t q_rd = 0, q_rl = 0, t_rd = 0, t_gr = 0, last = wall();
    bool gr_wait = false;
    bool started = false;
    while (q_rl < total || w_p < NP) {
        if (w_p < NP && (w_p == 0 || q_rd > 0 || total == 0)) {  // (W_r^T piece w_p, after my first x read)
            for (uint32_t k = w_p * BK; k < (w_p + 1) * BK; ++k) {
                for (uint32_t e = 0; e < EPC; ++e) {
                    noc_async_read_page((c * EPC + e) * KT + k, wt, wl1 + (k * EPC + e) * tile);
                }
            }
            noc_async_read_barrier();
            cb_push_back(w_cb, BK * EPC);
            if (++w_p == NP) {
                stamps[5] = wall();
            }
        }
        {
            const uint32_t now = wall();
            t_gr += gr_wait ? now - last : 0;
            last = now;
            gr_wait = false;
        }
        if (q_rd < total && cb_pages_reservable_at_back(xs_cb, BK)) {
            const uint32_t s = stream(q_rd), r = R0 + s / NP, col = (s % NP) * piece;
            cb_reserve_back(xs_cb, BK);
            uint32_t l1 = get_write_ptr(xs_cb);
            for (uint32_t j = 0; j < 32; ++j, l1 += piece) {
                noc_async_read(xs.get_noc_addr(r * 32 + j, col), l1, piece);
            }
            const uint32_t a = wall();
            noc_async_read_barrier();
            t_rd += wall() - a;
            last += wall() - a;
            cb_push_back(xs_cb, BK);
            ++q_rd;
        }
        if (q_rl < q_rd && !started) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(6)), 1);
            started = true;
        }
        if (q_rl < q_rd && cb_pages_available_at_front(tz_cb, BK)) {
            const uint32_t s = stream(q_rl);
            invalidate_l1_cache();
            gr_wait = s >= NR && *grant < s + 1;
            if (!gr_wait) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bscr) = s + 1;
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    get_read_ptr(tz_cb), get_noc_addr(hx, hy, relay + (s % NR) * BK * tile), BK * tile);
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(
                    bscr, get_noc_addr(hx, hy, rflags + (s % NR) * 16), 4);
                noc_async_posted_writes_flushed();  // (the piece's CB slot and the scratch word are reused)
                cb_pop_front(tz_cb, BK);
                ++q_rl;
            }
        }
    }
    noc_async_posted_writes_flushed();
    stamps[6] = t_rd;
    stamps[7] = t_gr;
}
