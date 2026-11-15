// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert store-and-forward forwarder, BRISC (NOC0; NOC1 with RCfg.swap_noc). Loads this ring's order table
// (wr_seq.hpp) into CB 2 for both RISCs, then runs two independent cursors over it in one non-blocking loop:
//   INJ:       reads each packet's chunk from this chip's DRAM into CB 0 (RD_DEPTH chunks in flight, one transaction
//              id each; slices stored chunk-contiguous, ws_fab.slice_layout) for the NCRISC to send;
//   FWD/TERM:  once the packet is in staging slot k % S (arrival count > k), writes it into its group core's ring slot
//              (the slice's first packet waits until the core freed the slot's previous expert: credit word >= n - 1,
//              fanned out by wr_relay); after the slice's last packet it writes the slot's arrival count.
//              The landed count goes to this core's W_LANDED word, which the NCRISC folds into the staging credits.
// The cursors are independent so that the owner's reads run ahead of (and never wait on) the packets passing through.
// RT: 0 ring index (table page), 1 table length, 2 table addr, 3 slices addr, 4 this core's NoC xy (x | y << 8),
//     5.. NC targets (c | x << 8 | y << 16)
// CT: 0 NEXP, 1 KT, 2 KTD, 3 DCOLS, 4 CHT, 5 G, 6 GC, 7 NCHIP, 8 S, 9 words, 10 wgu_ring, 11 wd_ring, 12 warr,
//     13 icred (group slot credit word c), 14 TL (table page entries), then TensorAccessorArgs(slices),
//     TensorAccessorArgs(table)
#include "api/dataflow/dataflow_api.h"
#include "wr_seq.hpp"

void kernel_main() {
    constexpr uint32_t NEXP = get_compile_time_arg_val(0), KT = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2), DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t CHT = get_compile_time_arg_val(4), G = get_compile_time_arg_val(5);
    constexpr uint32_t GC = get_compile_time_arg_val(6), NCHIP = get_compile_time_arg_val(7);
    constexpr uint32_t S = get_compile_time_arg_val(8), words = get_compile_time_arg_val(9);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(10), wd_ring = get_compile_time_arg_val(11);
    constexpr uint32_t warr = get_compile_time_arg_val(12), icred = get_compile_time_arg_val(13);
    constexpr uint32_t TL = get_compile_time_arg_val(14);
    constexpr auto wargs = TensorAccessorArgs<15>();
    constexpr auto targs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS, WTOT = WGU + WDN;
    static_assert(WGU % CHT == 0 && WTOT % CHT == 0);
    constexpr uint32_t NCH = WTOT / CHT;
    constexpr uint32_t D = RD_DEPTH, SCB = SCB_CHUNKS;  // (D <= SCB, D <= 15)
    static_assert(NEXP < (1u << 18) && NCHIP > 0);

    const uint32_t ring = get_arg_val<uint32_t>(0), len = get_arg_val<uint32_t>(1);
    const uint32_t tb = get_tile_size(0);
    const uint32_t CHB = CHT * tb;
    const auto tab_acc = TensorAccessor(targs, get_arg_val<uint32_t>(2), TL * 4);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(3), tb);
    const uint32_t mxy = get_arg_val<uint32_t>(4);
    constexpr uint32_t TGT = 5;
    const uint32_t stage = get_write_ptr(1);
    const uint32_t inj = get_write_ptr(0);
    const uint32_t tab = get_write_ptr(2);

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint32_t tstart = wall();
    noc_async_read(tab_acc.get_noc_addr(ring), tab, TL * 4);
    noc_async_read_barrier();
    noc_inline_dw_write(get_noc_addr(mxy & 0xFF, mxy >> 8, words + wr::W_TABLE * 16), 1);
    volatile tt_l1_ptr uint32_t* T = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tab);
    volatile tt_l1_ptr uint32_t* arr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(words + wr::W_ARR * 16);
    const uint64_t landed_dst = get_noc_addr(mxy & 0xFF, mxy >> 8, words + wr::W_LANDED * 16);

    uint32_t t_arr = 0, t_gc = 0, t_land = 0, t_inj = 0;
    uint32_t ii = 0, il = 0, issued = 0, pushed = 0, k = 0;
    auto next_inj = [&]() {
        while (ii < len && (T[ii] & 3) != wr::INJ) {
            ++ii;
        }
    };
    auto next_land = [&]() {
        while (il < len && (T[il] & 3) == wr::INJ) {
            ++il;
        }
    };
    next_inj();
    next_land();
#ifdef WS_DIAG
    // progress counters for a hang post-mortem (CB 5 / 6 pushed / popped): CB 5 pushed = landing table index, popped
    // = packets landed; CB 6 pushed = INJ table index, popped = 256 (slice starts landed) + the slot credit seen by
    // the next slice start
    uint32_t dg[4] = {0, 0, 0, 0}, cseen = 0, nsl = 0;
    auto diag = [&]() {
        const uint32_t v[4] = {il, k, ii, 256 * nsl + cseen};
        for (uint32_t q = 0; q < 4; ++q) {
            if (v[q] > dg[q]) {
                if (q % 2 == 0) {
                    cb_push_back(5 + q / 2, v[q] - dg[q]);
                } else {
                    cb_pop_front(5 + q / 2, v[q] - dg[q]);
                }
                dg[q] = v[q];
            }
        }
    };
#endif
    while (ii < len || il < len || pushed < issued) {
        const uint32_t a0 = wall();
#ifdef WS_DIAG
        diag();
#endif
        bool did = false;
        // INJ: retire the oldest read once it is done (or when D are in flight), issue the next one when CB 0 has room
        if (pushed < issued) {
            const uint32_t trid = 1 + pushed % D;
            if (issued - pushed >= D || ii >= len ||
                (noc_cmd_buf_ready(noc_index, read_cmd_buf) &&
                 ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid))) {
                noc_async_read_barrier_with_trid(trid);
                cb_push_back(0, CHT);
                ++pushed;
                did = true;
            }
        }
        if (ii < len && issued - pushed < D && cb_pages_reservable_at_back(0, (issued - pushed + 1) * CHT)) {
            cb_reserve_back(0, (issued - pushed + 1) * CHT);
            const wr::Pkt p = wr::decode(T[ii]);
            // chunk q: pages (q / 8) CHT 8 + q % 8 + 8 k (WS_OWN_SLICES: only the slices this chip owns, at local
            // expert ws_sf.local_expert)
#if defined(WS_OWN_SLICES) && defined(WS_LANE_OWNERS)
            const uint32_t q = (((p.e / G / NCHIP) * G + p.e % G) * GC + p.c % GC) * NCH + p.u;
#elif defined(WS_OWN_SLICES)
            const uint32_t q = ((p.e / NCHIP) * GC + p.c % GC) * NCH + p.u;
#else
            const uint32_t q = (p.e * GC + p.c % GC) * NCH + p.u;
#endif
            noc_async_read_set_trid(1 + issued % D);
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), inj + (issued % SCB) * CHB, CHB);
            noc_async_read_set_trid(0);
            ++issued;
            ++ii;
            next_inj();
            did = true;
        }
        if (did) {
            t_inj += wall() - a0;
        }
        // FWD / TERM: land the next incoming packet once it is here and its group slot is free
        if (il < len) {
            const uint32_t a1 = wall();
            const wr::Pkt p = wr::decode(T[il]);
            const uint32_t n = p.e / G;
            invalidate_l1_cache();
            if (*arr < k + 1) {
                if (!did) {
                    t_arr += wall() - a0;
                }
                continue;
            }
#ifdef WS_DIAG
            cseen = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + p.c * 16);
#endif
            if (p.u == 0 && n >= 2 && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(icred + p.c * 16) < n - 1) {
                if (!did) {
                    t_gc += wall() - a0;
                }
                continue;
            }
#ifdef WS_DIAG
            if (p.u == 0) {
                ++nsl;
                cseen = 0;
            }
#endif
            const uint32_t t = get_arg_val<uint32_t>(TGT + p.c);
            const uint32_t tx = (t >> 8) & 0xFF, ty = (t >> 16) & 0xFF;
            const uint32_t u0 = p.u * CHT;
            const uint32_t dst =
                u0 < WGU ? wgu_ring + ((n & 1) * WGU + u0) * tb : wd_ring + ((n & 1) * WDN + u0 - WGU) * tb;
            noc_async_write(stage + (k % S) * CHB, get_noc_addr(tx, ty, dst), CHB);
            if (p.u == NCH - 1) {
                // (the slot's arrival word as an absolute count, n / 2 + 1 experts of this parity: one lander writes a
                // core's slot at a time and the counts only grow, so a plain write does and no NoC atomic goes to a
                // busy group core)
                noc_async_write_barrier();
                noc_inline_dw_write(get_noc_addr(tx, ty, warr + (n & 1) * 16), n / 2 + 1);
            } else {
                noc_async_writes_flushed();
            }
            ++k;
            noc_inline_dw_write(landed_dst, k);
            ++il;
            next_land();
            t_land += wall() - a1;
        }
    }
    noc_async_full_barrier();
    volatile tt_l1_ptr uint32_t* sst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SST_ADDR);
    sst[16] = t_arr;
    sst[17] = t_gc;
    sst[18] = t_land;
    sst[19] = t_inj;
    sst[20] = wall() - tstart;
    sst[21] = k;
    sst[22] = issued;
}
