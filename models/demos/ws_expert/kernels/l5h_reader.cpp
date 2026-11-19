// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups, h by the group hub's multicast (ws_lane5.py, L5Cfg.hmc): the
// group NCRISC (NOC1), carrying every unicast and DRAM transfer of the core so that inside a group's rectangle NOC0 holds
// only the hub's multicasts (l5h_hub.cpp, the BRISC of core 0 at the rectangle's NOC0 multicast start corner); the
// other cores run no BRISC kernel. Core (C, Y) of a group, j = 5 C + Y. Loads the group's slot table and the counts
// (l5_writer.cpp's start), then a non-blocking loop over l5_reader.cpp's QW / QF / QR (weights), QB (x ring), QL
// (landing, credits to every other core of the column), QY (outputs), l5_writer.cpp's Q1 (x gather), QX (x rows to the
// row's 5 cores), QO (own experts from DRAM), the partials to all 4 owners of the column (QP), and:
//   QA (owners): h tile of row q -> slot q % NS of the hub's h ring at index 4 C + Y, then its flag word (slot, 4 C + Y)
//      = q + 1, once the hub freed row q - NS (HCRED word 0: rows the hub both consumed and multicast).
//   QH: push row q (CB 2, 20 tiles) once the hub's multicast flag (HCOLF word 5 s, s = q % NS) reads >= q + 1; when
//      compute pops h rows, the count into HCONS word j at the hub.
//   QZ (hub, j 0): rows freed = min(own consumed, multicasts sent (scratch word 3, by the hub's BRISC)) -> HCRED word 0
//      of the 20 owners.
// WL5_HMC2 (L5Cfg.hmc 2): the BRISC (l5h_writer.cpp) keeps the transfers that stay inside the rectangle on NOC0 (x rows
// east, partials and landing credits south, the hub's multicasts); this RISC sends the x rows west (from the row the
// BRISC publishes, XAVAIL / XADDR, then XDONE), the partials to the owners above (PAVAIL / PDONE) and the landing
// credits to the rows above.
// CT: as l5_reader.cpp (0-36, TensorAccessorArgs(out)), then x ring, x tile, table page bytes, RMAX, x row bytes,
//     idx L1, NEXP, PKE, RD, HCONS, then TensorAccessorArgs(x), (idx), (counts), (table), (own slices).
// RT: as l5_reader.cpp (0-19, the mux client args 20-53), then 54 x addr, 55 idx addr, 56 counts addr, 57 K0,
//     58 table addr, 59 own slices addr, 60 CH0, 61 destination lists addr (WL5_Y_RM; CT TensorAccessorArgs(dest)
//     after the own slices').
// WL5_Y_RM: row-major result rows - each down row tile's rows (CB 17, pack-untilized) to their rows in the expert's
// destination list, at this core's columns, posted, fenced (WL5_Y_FENCE: the result buffer's spare rows). WL5_YT (the
// staging slot's L1 address): the rows come from the BRISC's transposed copy (l5h_writer.cpp QT; YREADY / YDONE in
// scratch words 12 / 13) instead of CB 17. WL5_YC: the rows go to the group's row collectors instead (QY below); RT
// 62 .. 62 + WL5_YC - 1 their NoC coordinates (x | y << 8). WL5_OSIDE (no WL5_YC): RT 62 this group's side (0 west,
// 1 east): result row d at page (d >> 2) 8 + 4 side + (d & 3), the side's DRAM column.
// WL5_OSIDE 2: three banks a side, row d at page (d / 3) 8 + 4 side + 1 + d % 3.
// WL5_ODROP (with WL5_OSIDE; bit side): no row writes on that side (diagnostics: the rows' NoC traffic).
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t land_cb = tt::CBIndex::c_4;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t p_cb = tt::CBIndex::c_18;

void kernel_main() {
    constexpr uint32_t KSL = get_compile_time_arg_val(0), DCOLS = get_compile_time_arg_val(1);
    constexpr bool OWN = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t NXR = get_compile_time_arg_val(3);
    constexpr uint32_t xflags = get_compile_time_arg_val(4), xcred = get_compile_time_arg_val(5);
    constexpr uint32_t D = get_compile_time_arg_val(6);
    constexpr uint32_t land = get_compile_time_arg_val(7), lflags = get_compile_time_arg_val(8);
    constexpr uint32_t lcred = get_compile_time_arg_val(9), ptile = get_compile_time_arg_val(10);
    constexpr uint32_t scr = get_compile_time_arg_val(11), KTD = get_compile_time_arg_val(12);
    constexpr uint32_t NS = get_compile_time_arg_val(13), hflags = get_compile_time_arg_val(14);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(15), table_l1 = get_compile_time_arg_val(16);
    constexpr uint32_t warr_addr = get_compile_time_arg_val(17);
    constexpr uint32_t ORMAX = get_compile_time_arg_val(18), HT = get_compile_time_arg_val(19);
    constexpr uint32_t wring = get_compile_time_arg_val(20);
    constexpr uint32_t cred_addr = get_compile_time_arg_val(21), freew_addr = get_compile_time_arg_val(22);
    constexpr uint32_t CHT = get_compile_time_arg_val(23), wtile = get_compile_time_arg_val(24);
    constexpr uint32_t PD = get_compile_time_arg_val(25), pring = get_compile_time_arg_val(26);
    constexpr uint32_t hcred = get_compile_time_arg_val(27), K = get_compile_time_arg_val(28);
    constexpr uint32_t hcolf = get_compile_time_arg_val(29), hring = get_compile_time_arg_val(30);
    constexpr uint32_t htile = get_compile_time_arg_val(31);
    constexpr uint8_t mux_nbuf = get_compile_time_arg_val(32);
    constexpr size_t mux_bufsz = get_compile_time_arg_val(33);
    constexpr size_t mux_status = get_compile_time_arg_val(34);
    constexpr size_t mux_term = get_compile_time_arg_val(35);
    constexpr uint32_t mux_clients = get_compile_time_arg_val(36);
    constexpr auto oargs = TensorAccessorArgs<37>();
    constexpr uint32_t E0 = oargs.next_compile_time_args_offset();
    constexpr uint32_t xring = get_compile_time_arg_val(E0), xtile = get_compile_time_arg_val(E0 + 1);
    constexpr uint32_t TB = get_compile_time_arg_val(E0 + 2), RMAX = get_compile_time_arg_val(E0 + 3);
    constexpr uint32_t xrow = get_compile_time_arg_val(E0 + 4), idx_l1 = get_compile_time_arg_val(E0 + 5);
    constexpr uint32_t NEXP = get_compile_time_arg_val(E0 + 6), PKE = get_compile_time_arg_val(E0 + 7);
    constexpr uint32_t RD = get_compile_time_arg_val(E0 + 8), hcons = get_compile_time_arg_val(E0 + 9);
    constexpr auto xargs = TensorAccessorArgs<E0 + 10>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr auto targs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto wargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
#if defined(WL5_Y_RM)
    constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
#endif
    static_assert(RD >= 1 && RD <= 8);
#if defined(WL5_GATHER0)
    constexpr bool GATHER0 = true;  // (the BRISC gathers x on NOC0: diagnostic)
#else
    constexpr bool GATHER0 = false;
#endif
    constexpr uint32_t GTRID = 15;  // (the x gather's reads; the own chunk reads use 1 .. RD)
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR, NH = NOWN * NCOL;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;
    static_assert(WGU % CHT == 0 && WDN % CHT == 0);
    static_assert(K <= 7);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    const uint32_t DS = get_arg_val<uint32_t>(5);
    const uint16_t my_mesh = get_arg_val<uint32_t>(6);
    // (RT 7 / 8: the direction-d neighbour's chip id | (this lane's core j there: x | y << 8) << 8 - a chip may hold
    // a lane on another rectangle than its neighbours, L5Cfg.place)
    const uint8_t nchip[2] = {(uint8_t)get_arg_val<uint32_t>(7), (uint8_t)get_arg_val<uint32_t>(8)};
    const uint32_t nbx[2] = {(get_arg_val<uint32_t>(7) >> 8) & 0xFF, (get_arg_val<uint32_t>(8) >> 8) & 0xFF};
    const uint32_t nby[2] = {get_arg_val<uint32_t>(7) >> 16, get_arg_val<uint32_t>(8) >> 16};
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(10 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(15 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    constexpr uint32_t RX = 20 + 2 * 17;  // (the extra RT args, after both mux connections)
    const uint32_t xaddr = get_arg_val<uint32_t>(RX), iaddr = get_arg_val<uint32_t>(RX + 1);
    const uint32_t caddr = get_arg_val<uint32_t>(RX + 2), K0 = get_arg_val<uint32_t>(RX + 3);
    const uint32_t CH0 = get_arg_val<uint32_t>(RX + 6);
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);
    const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(RX + 4), TB);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(RX + 5), wtile);
    const uint32_t hubx = cx[0], huby = ry[0];
    constexpr uint32_t GC = NR * NCOL;

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    noc_async_read_page(0, cs, counts_l1 + 64);
    noc_async_read_page(g, ts, table_l1);
    noc_async_read_barrier();
    counts[0] = 1;
    volatile tt_l1_ptr uint32_t* tab = W(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

    // ---- mux connections (index d: to the next chip in direction d), headers. WL5_ONEMUX (a 2-chip cycle: both
    // directions lead to the same chip over the same link, and a router direction takes one worker connection): one
    // mux, both directions' packets and credits through it (the RT args of direction 1 are present, unused)
#ifdef WL5_ONEMUX
    constexpr uint32_t NMUX = 1;
#else
    constexpr uint32_t NMUX = 2;
#endif
#define MUXD(d) mux[NMUX == 1 ? 0 : (d)]
    tt::tt_fabric::WorkerToFabricMuxSender<mux_nbuf> mux[2];
    bool master[2];
    uint32_t term_sync[2], tmx[2], tmy[2];
    uint8_t muxx[2], muxy[2];
    for (uint32_t d = 0; d < NMUX; ++d) {
        size_t a_ = 20 + 17 * d;
        a_++;  // connection valid
        master[d] = get_arg_val<uint32_t>(a_++) != 0;
        muxx[d] = get_arg_val<uint32_t>(a_++);
        muxy[d] = get_arg_val<uint32_t>(a_++);
        const size_t ch_base = get_arg_val<uint32_t>(a_++), conn_info = get_arg_val<uint32_t>(a_++);
        const size_t handshake = get_arg_val<uint32_t>(a_++), flow = get_arg_val<uint32_t>(a_++);
        const size_t bufidx = get_arg_val<uint32_t>(a_++);
        const uint8_t chan = get_arg_val<uint32_t>(a_++);
        term_sync[d] = get_semaphore(get_arg_val<uint32_t>(a_++));
        const uint32_t l_status = get_semaphore(get_arg_val<uint32_t>(a_++));
        const uint32_t l_flow = get_semaphore(get_arg_val<uint32_t>(a_++));
        const uint32_t l_teardown = get_semaphore(get_arg_val<uint32_t>(a_++));
        const uint32_t l_bufidx = get_semaphore(get_arg_val<uint32_t>(a_++));
        tmx[d] = get_arg_val<uint32_t>(a_++);
        tmy[d] = get_arg_val<uint32_t>(a_++);
#ifdef WL5_HO_BUFSZ
        // (one-way lanes: direction 1 is a header-only channel of the partner lane's mux, for the credits)
        const size_t bufsz = d == 1 ? WL5_HO_BUFSZ : mux_bufsz;
#else
        const size_t bufsz = mux_bufsz;
#endif
        mux[d] = tt::tt_fabric::build_connection_to_fabric_endpoint<mux_nbuf>(
            muxx[d],
            muxy[d],
            chan,
            mux_nbuf,
            bufsz,
            ch_base,
            conn_info,
            handshake,
            flow,
            bufidx,
            l_flow,
            l_teardown,
            l_bufidx);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(muxx[d], muxy[d], mux_status, l_status);
        tt::tt_fabric::fabric_client_connect(mux[d]);
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hcr = PacketHeaderPool::allocate_header();

    // ---- slots: current occupant (its table entry cached in registers), packets sent a direction, computed
    volatile tt_l1_ptr uint32_t* warr = W(warr_addr);
    volatile tt_l1_ptr uint32_t* cred = W(cred_addr);
    volatile tt_l1_ptr uint32_t* freew = W(freew_addr);
    uint32_t occ[K] = {}, sent[K][2] = {}, cur_e[K], cur_fwd[K], cur_nk[K], cur_dst[K][2], base[K];
    bool live[K], done_c[K] = {};
    auto load = [&](uint32_t s) {
        live[s] = occ[s] < tab[1 + s];
        base[s] = PK * occ[s];
        if (live[s]) {
            volatile tt_l1_ptr uint32_t* en = tab + tab[1 + K + s] + 4 * occ[s];
            const uint32_t w0 = en[0];
            cur_e[s] = w0 & 0xFFFF;
            cur_fwd[s] = (w0 >> 18) & 3;
            cur_nk[s] = (w0 >> 20) & 3;
            cur_dst[s][0] = en[1];
            cur_dst[s][1] = en[2];
        } else {
            cur_e[s] = 0xFFFF;
            cur_fwd[s] = cur_nk[s] = 0;
        }
    };
    for (uint32_t s = 0; s < K; ++s) {
        load(s);
    }
    auto slot_addr = [&](uint32_t s, uint32_t u) {  // packet u of slot s in this core's L1 (same on every chip)
        return u < GU_PK ? wring + (s * WGU + u * CHT) * wtile
                         : wring + (K * WGU + s * WDN + (u - GU_PK) * CHT) * wtile;
    };

    // ---- QW
    uint32_t w_n = 0, w_dn = 0, freed = 0, gu_pages = 0, dn_pages = 0;
    const uintptr_t gu_ack = (uintptr_t)get_cb_tiles_acked_ptr(w_cb), dn_ack = (uintptr_t)get_cb_tiles_acked_ptr(wd_cb);
    uint16_t gu_last = (uint16_t)reg_read(gu_ack), dn_last = (uint16_t)reg_read(dn_ack);
    auto qw = [&]() -> bool {
        bool did = false;
        if (w_n < nexp_g) {
            const uint32_t o = order[w_n], s = o >> 16;
            if (cur_e[s] == (o & 0xFFFF) && warr[s * 4] >= base[s] + GU_PK && cb_pages_reservable_at_back(w_cb, WGU)) {
                cb_reserve_back(w_cb, WGU);
                cb_push_back(w_cb, WGU);
                ++w_n;
                did = true;
            }
        }
        if (w_dn < w_n) {
            const uint32_t o = order[w_dn], s = o >> 16;
            if (cur_e[s] == (o & 0xFFFF) && warr[s * 4] >= base[s] + PK && cb_pages_reservable_at_back(wd_cb, WDN)) {
                cb_reserve_back(wd_cb, WDN);
                cb_push_back(wd_cb, WDN);
                ++w_dn;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(gu_ack), b = (uint16_t)reg_read(dn_ack);
        gu_pages += (uint16_t)(a - gu_last);
        dn_pages += (uint16_t)(b - dn_last);
        gu_last = a;
        dn_last = b;
        uint32_t f = gu_pages / WGU;
        if (dn_pages / WDN < f) {
            f = dn_pages / WDN;
        }
        while (freed < f) {
            done_c[order[freed] >> 16] = true;
            ++freed;
            did = true;
        }
        return did;
    };
    // ---- QF / QR
    uint32_t rr = 0, n_fwd = 0, n_credit = 0;
    uint32_t why = 0;  // (WL5_WHY: what blocked the forwards when none went: 1 arrival, 2 credit, 4 mux space)
    auto qf = [&]() -> bool {
        why = 0;
        for (uint32_t t = 0; t < 2 * K; ++t) {
            const uint32_t s = (rr + t) % K, d = ((rr + t) / K) % 2;
            if (!((cur_fwd[s] >> d) & 1) || sent[s][d] >= PK) {
                continue;
            }
            const uint32_t u = sent[s][d];
            if (warr[s * 4] < base[s] + u + 1) {
                why |= 1;
                continue;
            }
            const uint32_t s2 = cur_dst[s][d] & 7, k2 = cur_dst[s][d] >> 3;
            if (u == 0 && cred[(d * K + s2) * 4] < k2) {
                why |= 2;
                continue;
            }
            if (!MUXD(d).edm_has_space_for_packet()) {
                why |= 4;
                continue;
            }
            noc_async_writes_flushed();  // (the previous send's header and payload out of L1)
            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &MUXD(d),
                hdr[d],
                nchip[d],
                my_mesh,
                slot_addr(s, u),
                CHT * wtile,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(nbx[d], nby[d], slot_addr(s2, u)),
                    get_noc_addr(nbx[d], nby[d], warr_addr + s2 * 16),
                    1,
                    false});
            ++sent[s][d];
            ++n_fwd;
            rr = rr + t + 1;
            return true;
        }
        return false;
    };
    auto qr = [&]() -> bool {
        bool did = false;
        for (uint32_t s = 0; s < K; ++s) {
            if (!live[s] || !done_c[s]) {
                continue;
            }
            const uint32_t fwd = cur_fwd[s];
            if (((fwd & 1) && sent[s][0] < PK) || ((fwd & 2) && sent[s][1] < PK)) {
                continue;
            }
            const uint32_t nk = cur_nk[s], next = occ[s] + 1;
            if (nk >= 2 && !MUXD(1 - (nk - 2)).edm_has_space_for_packet()) {
                continue;
            }
            noc_async_writes_flushed();  // (every forward of the slot has left L1 before it is refilled)
            if (nk == 1) {
                freew[s * 4] = next;
            } else if (nk >= 2) {
                const uint32_t d = nk - 2;
                fabric_unicast_noc_unicast_inline_write(
                    &MUXD(1 - d),
                    hcr,
                    nchip[1 - d],
                    my_mesh,
                    tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                        get_noc_addr(nbx[1 - d], nby[1 - d], cred_addr + (d * K + s) * 16), next});
                ++n_credit;
            }
            occ[s] = next;
            sent[s][0] = sent[s][1] = 0;
            done_c[s] = false;
            load(s);
            did = true;
        }
        return did;
    };
    auto slots_done = [&]() {
        for (uint32_t s = 0; s < K; ++s) {
            if (live[s]) {
                return false;
            }
        }
        return true;
    };

    // ---- QB: x ring pushes + credits to the row
    uint32_t b_s = 0, x_pages = 0, x_cred = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_rows && cb_pages_reservable_at_back(xr_cb, KSL)) {
            invalidate_l1_cache();
            if (*W(xflags + (b_s % NXR) * 16) >= b_s + 1) {
                cb_reserve_back(xr_cb, KSL);
                cb_push_back(xr_cb, KSL);
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_pages += (uint16_t)(a - x_last);
        x_last = a;
        if (x_pages / KSL > x_cred) {
            x_cred = x_pages / KSL;
            *W(xcred + C * 16) = x_cred;
            for (uint32_t t = 1; t < NCOL; ++t) {
                noc_inline_dw_write(get_noc_addr(cx[(C + t) % NCOL], ry[Y], xcred + C * 16), x_cred);
            }
            did = true;
        }
        return did;
    };

    // ---- QL (owners)
    uint32_t l_q = 0, l_pages = 0, l_cons = 0;
    const uintptr_t lacked = (uintptr_t)get_cb_tiles_acked_ptr(land_cb);
    uint16_t l_last = (uint16_t)reg_read(lacked);
    auto ql = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        bool did = false;
        if (l_q < total_rows && cb_pages_reservable_at_back(land_cb, LT)) {
            invalidate_l1_cache();
            bool all = true;
            for (uint32_t k = 0; k < NR && all; ++k) {
                all = *W(lflags + ((l_q % D) * NR + k) * 16) >= l_q + 1;
            }
            if (all) {
                cb_reserve_back(land_cb, LT);
                cb_push_back(land_cb, LT);
                ++l_q;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(lacked);
        l_pages += (uint16_t)(a - l_last);
        l_last = a;
        if (l_pages / LT > l_cons) {
            l_cons = l_pages / LT;
            *W(lcred + Y * 16) = l_cons;
#if defined(WL5_HMC2)
            for (uint32_t o = 0; o < Y; ++o) {  // (the rows below: the BRISC, NOC0)
#else
            for (uint32_t o = 0; o < NR; ++o) {
                if (o == Y) {
                    continue;
                }
#endif
                noc_inline_dw_write(get_noc_addr(cx[C], ry[o], lcred + Y * 16), l_cons);
            }
            did = true;
        }
        return did;
    };

#if defined(WL5_HMC2)
    // ---- QP: partials to the owners above this row (the BRISC sends to those at and below)
    const uint32_t nup = Y < NOWN ? Y : NOWN;
    uint32_t pq = 0, po = 0;
    auto qp = [&]() -> bool {
        if (pq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(scr + 4 * 16) < pq + 1) {  // PAVAIL
            return false;
        }
        if (po == 0) {
            *W(scr + 7 * 16) = pq + 1;
        }
        const uint32_t src = pring + (pq % PD) * NW * ptile, s = pq % D;
        bool did = false;
        while (po < nup) {
            if (pq >= D && *W(lcred + po * 16) < pq - D + 1) {
                return did;
            }
            noc_async_write(
                src + 2 * po * ptile, get_noc_addr(cx[C], ry[po], land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 7 * 16, get_noc_addr(cx[C], ry[po], lflags + (s * NR + Y) * 16), 4);
            ++po;
            did = true;
        }
        noc_async_writes_flushed();
        *W(scr + 5 * 16) = pq + 1;  // PDONE
        ++pq;
        po = 0;
        return true;
    };
#else
    // ---- QP: partials to the 4 owners of the column (this core included when it is one)
    uint32_t pq = 0, po = 0;
    auto qp = [&]() -> bool {
        if (pq >= total_rows || !cb_pages_available_at_front(p_cb, NW)) {
            return false;
        }
        if (po == 0) {
            *W(scr + 16) = pq + 1;
        }
        const uint32_t src = get_read_ptr(p_cb), s = pq % D;
        bool did = false;
        while (po < NOWN) {
            if (pq >= D) {
                invalidate_l1_cache();
                if (*W(lcred + po * 16) < pq - D + 1) {
                    return did;
                }
            }
            noc_async_write(
                src + 2 * po * ptile, get_noc_addr(cx[C], ry[po], land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 16, get_noc_addr(cx[C], ry[po], lflags + (s * NR + Y) * 16), 4);
            ++po;
            did = true;
        }
        noc_async_writes_flushed();
        cb_pop_front(p_cb, NW);
        ++pq;
        po = 0;
        return true;
    };

#endif
    // ---- QA (owners): h tile of row q -> the hub's ring slot q % NS, then its flag
    uint32_t a_q = 0;
    const uint32_t hidx = PC * C + Y;
    auto qa = [&]() -> bool {
        if constexpr (!OWN) {
            return false;
        }
        if (a_q >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
        if (a_q >= NS) {
            invalidate_l1_cache();
            if (*W(hcred) < a_q - NS + 1) {
                return false;
            }
        }
        const uint32_t s = a_q % NS;
        *W(scr + 6 * 16) = a_q + 1;
        noc_async_write(get_read_ptr(h_cb), get_noc_addr(hubx, huby, hring + (s * KTD + hidx) * htile), htile);
        noc_async_write(scr + 6 * 16, get_noc_addr(hubx, huby, hflags + (s * NH + hidx) * 16), 4);
        noc_async_writes_flushed();
        cb_pop_front(h_cb, 1);
        ++a_q;
        return true;
    };
    // ---- QH: h ring rows from the hub's multicast; consumed counts to the hub
    uint32_t h_q = 0, h_pages = 0, h_cons = 0;
    const uintptr_t hacked = (uintptr_t)get_cb_tiles_acked_ptr(hr_cb);
    uint16_t h_last = (uint16_t)reg_read(hacked);
    auto qh = [&]() -> bool {
        bool did = false;
        if (h_q < total_rows && cb_pages_reservable_at_back(hr_cb, KTD)) {
            invalidate_l1_cache();
            if (*W(hcolf + (h_q % NS) * NCOL * 16) >= h_q + 1) {
                cb_reserve_back(hr_cb, KTD);
                cb_push_back(hr_cb, KTD);
                ++h_q;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(hacked);
        h_pages += (uint16_t)(a - h_last);
        h_last = a;
        if (h_pages / KTD > h_cons) {
            h_cons = h_pages / KTD;
            if (J == 0) {
                *W(hcons) = h_cons;
            } else {
                noc_inline_dw_write(get_noc_addr(hubx, huby, hcons + J * 16), h_cons);
            }
            did = true;
        }
        return did;
    };
    // ---- QZ (hub): rows freed in the hub's ring -> every owner
    uint32_t z_sent = 0;
    auto qz = [&]() -> bool {
        if (J != 0 || z_sent >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        uint32_t f = *W(scr + 3 * 16);
        if (h_cons < f) {
            f = h_cons;
        }
        if (f <= z_sent) {
            return false;
        }
        *W(hcred) = f;
        for (uint32_t c = 0; c < NCOL; ++c) {
            for (uint32_t o = 0; o < NOWN; ++o) {
                if (c != 0 || o != 0) {
                    noc_inline_dw_write(get_noc_addr(cx[c], ry[o], hcred), f);
                }
            }
        }
        z_sent = f;
        return true;
    };

#if defined(WL5_Y_RM) && defined(WL5_YC)
    // ---- QY (WL5_YC: the group's WL5_YC row collectors, l5_ycoll.cpp): row tile n goes to collector n % WL5_YC, its
    // staging slot (n / WL5_YC) % WL5_YC_NSL (32 rows of HT 64 B at the weight ring's address there: the ring is sharded
    // on the group cores only, so that range is free on the collectors); this core's rows land at its columns (DS 64 B
    // into each row), posted, once the collector freed the slot (its count of freed row tiles, word c of WL5_YC_CRED
    // here); once the writes have left, + 1 on the slot's arrival word (WL5_YC_ARR + 16 s there: the same NoC and VC
    // as the rows, behind them) and CB 17's pages are freed. The collector writes each full row to its destination.
    constexpr uint32_t NCY = WL5_YC, NSL = WL5_YC_NSL;
    constexpr uint32_t seg = DCOLS * 64, ROWB = HT * 64, SLOTB = 32 * ROWB;
    uint32_t ycx[NCY];
    for (uint32_t i = 0; i < NCY; ++i) {
        ycx[i] = get_arg_val<uint32_t>(RX + 8 + i);
    }
    uint32_t drained = 0, d_n = 0, d_r = 0, y_state = 0;
    auto qy = [&]() -> bool {
        if (y_state == 1) {
            if (!ncrisc_noc_posted_writes_sent(noc_index)) {
                return false;
            }
            const uint32_t c = drained % NCY, s = (drained / NCY) % NSL;
            noc_semaphore_inc(get_noc_addr(ycx[c] & 0xFF, ycx[c] >> 8, WL5_YC_ARR + s * 16), 1);
            cb_pop_front(y_cb, DCOLS);
            ++d_r;
            ++drained;
            y_state = 0;
            return true;
        }
        if (drained >= total_rows) {
            return false;
        }
        while (d_r >= rows_of(d_n)) {
            ++d_n;
            d_r = 0;
        }
        const uint32_t c = drained % NCY, k = drained / NCY;
        invalidate_l1_cache();
        if (k >= *W(WL5_YC_CRED + c * 16) + NSL || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        const uint32_t m = counts[16 + (order[d_n] & 0xFFFF)] - d_r * 32;
        const uint32_t nr = m < 32 ? m : 32;
        uint32_t l1 = get_read_ptr(y_cb);
        const uint64_t base = get_noc_addr(ycx[c] & 0xFF, ycx[c] >> 8, wring + (k % NSL) * SLOTB + DS * 64);
        for (uint32_t r = 0; r < nr; ++r, l1 += seg) {
            noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(l1, base + r * ROWB, seg);
        }
        y_state = 1;
        return true;
    };
#elif defined(WL5_Y_RM)
    // ---- QY: each row of a down row tile to its destination row of the row-major result buffer, at this core's
    // columns (wl_reader.cpp's WL_Y_RM): a state machine (0: next row tile, 1: its 32 destinations requested, 2: its rows
    // written); the destinations (the expert's destination list, the token lists' layout) are read on their own
    // transaction id, the row writes are posted (fenced at the end: WL5_Y_FENCE), and CB 17's pages are freed once the
    // writes have left
    constexpr uint32_t Y_TRID = 0xD;  // (own chunk reads use 1 .. RD, the gather GTRID)
    static_assert(RD < Y_TRID && Y_TRID != GTRID);
    constexpr uint32_t seg = DCOLS * 64;  // a row's bytes at this core's columns
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(4), HT * 64);
    const auto dls = TensorAccessor(dargs, get_arg_val<uint32_t>(RX + 7), RMAX * 32 * 4);
    constexpr uint32_t dst_l1 = idx_l1 + 256;
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);
    uint32_t drained = 0, d_n = 0, d_r = 0, y_state = 0;
#if defined(WL5_OSIDE)
    const uint32_t oside = get_arg_val<uint32_t>(RX + 8) * 4;
#if WL5_OSIDE == 2
    auto orow = [oside](uint32_t d) { return (d / 3) * 8 + oside + 1 + d % 3; };
#else
    auto orow = [oside](uint32_t d) { return (d >> 2) * 8 + oside + (d & 3); };
#endif
#else
    auto orow = [](uint32_t d) { return d; };
#endif
#if defined(WL5_YSTATE)
    uint64_t ybank[NUM_DRAM_BANKS];  // (bank b's NoC address of the result buffer's first row at this core's columns)
    for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
        ybank[b] = os.get_noc_addr(b, DS * 64);  // (page b: bank b's first page)
    }
#endif
    auto qy = [&]() -> bool {
        if (y_state == 2) {
            if (!ncrisc_noc_posted_writes_sent(noc_index)) {
                return false;
            }
#if defined(WL5_YT)
            *W(scr + 13 * 16) = drained + 1;  // YDONE: the staging slot is free
#else
            cb_pop_front(y_cb, DCOLS);
#endif
            ++d_r;
            ++drained;
            y_state = 0;
            return true;
        }
        if (y_state == 1) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, Y_TRID)) {
                return false;
            }
            invalidate_l1_cache();
#if defined(WL5_YT)
            if (*W(scr + 12 * 16) < drained + 1) {  // YREADY: the BRISC staged this row tile
                return false;
            }
            uint32_t l1 = WL5_YT;
#else
            if (!cb_pages_available_at_front(y_cb, DCOLS)) {
                return false;
            }
            uint32_t l1 = get_read_ptr(y_cb);
#endif
            const uint32_t m = counts[16 + (order[d_n] & 0xFFFF)] - d_r * 32;
            const uint32_t nr = m < 32 ? m : 32;
#if defined(WL5_YSTATE)
            // (the row tile's rows grouped by DRAM bank: one write state a bank - its NoC coordinates, the size, posted
            // - then only the source and the bank-local address a row; nothing else uses the write command buffer in
            // between, and the next ordinary write reprograms it whole)
            for (uint32_t b = 0; b < NUM_DRAM_BANKS; ++b) {
                bool first = true;
                for (uint32_t r = 0; r < nr; ++r) {
                    const uint32_t pg = dst[r];
                    if (pg % NUM_DRAM_BANKS != b) {
                        continue;
                    }
                    const uint64_t a = ybank[b] + (uint64_t)((pg / NUM_DRAM_BANKS) * (HT * 64));
                    if (first) {
                        noc_async_write_one_packet_set_state<true>(a, seg);
                        first = false;
                    }
                    noc_async_write_one_packet_with_state<true>(l1 + r * seg, (uint32_t)a);
                }
            }
#else
#if defined(WL5_ODROP)
            if (((WL5_ODROP) >> (oside >> 2)) & 1) {
                y_state = 2;
                return true;
            }
#endif
            for (uint32_t r = 0; r < nr; ++r, l1 += seg) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(l1, os.get_noc_addr(orow(dst[r]), DS * 64), seg);
            }
#endif
            y_state = 2;
            return true;
        }
        if (drained >= total_rows) {
            return false;
        }
        while (d_r >= rows_of(d_n)) {
            ++d_n;
            d_r = 0;
        }
        noc_async_read_set_trid(Y_TRID);
        noc_async_read(dls.get_noc_addr(order[d_n] & 0xFFFF) + d_r * 128, dst_l1, 128);
        noc_async_read_set_trid(0);
        y_state = 1;
        return true;
    };
#else
    // ---- QY: every expert's down rows, packed in the chip's order from tile row g ORMAX
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(4), obytes);
    uint32_t drained = 0;
    auto qy = [&]() -> bool {
        if (drained >= total_rows || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        const uint32_t row = g * ORMAX + drained;
        uint32_t l1 = get_read_ptr(y_cb);
        for (uint32_t t = 0; t < DCOLS; ++t, l1 += obytes) {
#if defined(WL5_OAFF)
            // (bank affinity: the group's tiles only in the 4 DRAM banks of its side - channels 0-3 at NoC x 0,
            // 4-7 at x 9 - so that its output writes do not cross the chip: logical tile q -> page 8 (q / 4) + q % 4 + 4 h)
            const uint32_t q = row * HT + DS + t;
            noc_async_write_page(8 * (q >> 2) + (q & 3) + 4 * ((WL5_OAFF >> g) & 1), os, l1);
#else
            noc_async_write_page(row * HT + DS + t, os, l1);
#endif
        }
        noc_async_writes_flushed();
        cb_pop_front(y_cb, DCOLS);
        ++drained;
        return true;
    };

#endif
    // ---- Q1: x gather (l5_writer.cpp's, reads with transaction id GTRID)
    constexpr uint32_t xrm_cb = tt::CBIndex::c_7, xst_cb = tt::CBIndex::c_8;
    const uint32_t koff = K0 * 64;
    uint32_t g_n = 0, g_r = 0, g_q = 0, g_state = 0;
    auto q1 = [&]() -> bool {
        if (g_state == 1) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, GTRID)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* idx = W(idx_l1);
            cb_reserve_back(xrm_cb, KSL);
            uint32_t l1 = get_write_ptr(xrm_cb);
            noc_async_read_set_trid(GTRID);
            for (uint32_t i = 0; i < 32; ++i, l1 += KSL * 64) {
                noc_async_read(xs.get_noc_addr(idx[i]) + koff, l1, KSL * 64);
            }
            noc_async_read_set_trid(0);
            g_state = 2;
            return true;
        }
        if (g_state == 2) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, GTRID)) {
                return false;
            }
            cb_push_back(xrm_cb, KSL);
            ++g_r;
            ++g_q;
            g_state = 0;
            return true;
        }
        while (true) {
            while (g_n < nexp_g && g_r >= rows_of(g_n)) {
                ++g_n;
                g_r = 0;
            }
            if (g_n >= nexp_g || g_q % NCOL == C) {
                break;
            }
            ++g_r;
            ++g_q;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, KSL)) {
            return false;
        }
        noc_async_read_set_trid(GTRID);
        noc_async_read(is.get_noc_addr(order[g_n] & 0xFFFF) + g_r * 32 * 4, idx_l1, 32 * 4);
        noc_async_read_set_trid(0);
        g_state = 1;
        return true;
    };
#if defined(WL5_HMC2)
    // ---- QX: own rows to the cores of the row west of this column (the BRISC sends the rest, NOC0 east)
    uint32_t xq = C;
    auto qx = [&]() -> bool {
        if (xq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(scr + 8 * 16) < xq + 1) {  // XAVAIL
            return false;
        }
        if (xq >= NXR) {
            for (uint32_t t = 0; t < C; ++t) {
                if (*W(xcred + t * 16) < xq - NXR + 1) {
                    return false;
                }
            }
        }
        const uint32_t dst = xring + (xq % NXR) * KSL * xtile, fl = xflags + (xq % NXR) * 16;
        const uint32_t src = *W(scr + 9 * 16);  // XADDR
        *W(scr + 11 * 16) = xq + 1;
        for (uint32_t tc = C; tc-- > 0;) {
            noc_async_write(src, get_noc_addr(cx[tc], ry[Y], dst), KSL * xtile);
            noc_async_write(scr + 11 * 16, get_noc_addr(cx[tc], ry[Y], fl), 4);
        }
        noc_async_writes_flushed();
        *W(scr + 10 * 16) = xq + 1;  // XDONE
        xq += NCOL;
        return true;
    };
#else
    // ---- QX: own rows to the 5 cores of the row
    uint32_t xq = C;
    auto qx = [&]() -> bool {
        if (xq >= total_rows || !cb_pages_available_at_front(xst_cb, KSL)) {
            return false;
        }
        if (xq >= NXR) {
            invalidate_l1_cache();
            for (uint32_t t = 0; t < NCOL; ++t) {
                if (*W(xcred + t * 16) < xq - NXR + 1) {
                    return false;
                }
            }
        }
        const uint32_t dst = xring + (xq % NXR) * KSL * xtile, fl = xflags + (xq % NXR) * 16;
        *W(scr) = xq + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        for (uint32_t t = 0; t < NCOL; ++t) {
            const uint32_t tc = (C + NCOL - t) % NCOL;  // (westwards first: NOC1's direction)
            noc_async_write(src, get_noc_addr(cx[tc], ry[Y], dst), KSL * xtile);
            noc_async_write(scr, get_noc_addr(cx[tc], ry[Y], fl), 4);
        }
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, KSL);
        xq += NCOL;
        return true;
    };
#endif
    // ---- QO: own experts' weights from DRAM into their slots (l5_writer.cpp's)
    auto entry = [&](uint32_t s, uint32_t k) { return tab + tab[1 + K + s] + 4 * k; };
    uint32_t o_s = 0, o_k[K] = {}, o_issued = 0, o_landed = 0, n_own = 0;
    bool o_busy = false;
    auto next_own = [&]() {
        invalidate_l1_cache();
        for (uint32_t t = 0; t < K; ++t) {
            const uint32_t s = (o_s + t) % K;
            while (o_k[s] < tab[1 + s] && ((entry(s, o_k[s])[0] >> 16) & 3) != 1) {
                ++o_k[s];
            }
            if (o_k[s] < tab[1 + s] && freew[s * 4] >= o_k[s]) {
                o_s = s;
                return true;
            }
        }
        return false;
    };
    auto qo = [&]() -> bool {
        bool did = false;
        if (!o_busy) {
            if (!next_own()) {
                return false;
            }
            o_busy = true;
            o_issued = o_landed = 0;
        }
        const uint32_t s = o_s, k = o_k[s];
        if (o_landed < o_issued) {
            const uint32_t trid = 1 + o_landed % RD;
            if (ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                ++o_landed;
                warr[s * 4] = PK * k + o_landed;
                did = true;
            }
        }
        if (o_issued < PK && o_issued - o_landed < RD) {
            const uint32_t u = o_issued;
            const uint32_t q = entry(s, k)[3] * PKE + CH0 + u;
            noc_async_read_set_trid(1 + u % RD);
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), slot_addr(s, u), CHT * wtile);
            noc_async_read_set_trid(0);
            ++o_issued;
            did = true;
        }
        if (o_landed >= PK) {
            o_busy = false;
            ++o_k[s];
            ++n_own;
            o_s = (s + 1) % K;
        }
        return did;
    };
    auto qo_done = [&]() {
        for (uint32_t s = 0; s < K; ++s) {
            if (o_k[s] < tab[1 + s]) {
                return false;
            }
        }
        return !o_busy;
    };

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint32_t t_start = wall();
    uint32_t tq[7] = {0, 0, 0, 0, 0, 0, 0}, tb[6] = {0, 0, 0, 0, 0, 0};
#if defined(WL5_WHY)
    uint32_t twhy[5] = {0, 0, 0, 0, 0}, t_first = 0, t_last = 0;
#endif
    while (true) {
        bool did[7], dib[6];
        uint32_t t[8], u[7];
        t[0] = wall();
        invalidate_l1_cache();
        did[0] = qw() | qf() | qr();
        t[1] = wall();
        did[1] = qb();
        t[2] = wall();
        did[2] = ql();
        t[3] = wall();
        did[3] = qp();
        t[4] = wall();
        did[4] = qh() | qz();
        t[5] = wall();
        did[5] = qy();
        t[6] = wall();
        did[6] = false;
        t[7] = t[6];
        u[0] = t[7];
        dib[0] = qx();
        u[1] = wall();
#if defined(WL5_GATHER0)
        dib[1] = false;
#else
        dib[1] = q1();
#endif
        u[2] = wall();
        dib[2] = false;
        u[3] = u[2];
        dib[3] = false;
        u[4] = u[3];
        dib[4] = qa();
        u[5] = wall();
        dib[5] = qo();
        u[6] = wall();
        bool any = false;
#if defined(WL5_WHY)
        if (!did[0] && !dib[5]) {  // (no forward, no own read: charge the iteration to what blocked the forwards)
            const uint32_t dt = u[6] - t[0];
            bool rpend = false;  // (a slot sent all, waiting to be refilled: compute or its mux)
            for (uint32_t s = 0; s < K; ++s) {
                rpend |= live[s] && !(((cur_fwd[s] & 1) && sent[s][0] < PK) || ((cur_fwd[s] & 2) && sent[s][1] < PK));
            }
            const uint32_t k = (why & 4) ? 3 : (why & 2) ? 2 : (why & 1) ? 1 : rpend ? 4 : 0;
            twhy[k] += dt;
        }
        if (did[0]) {
            if (n_fwd && !t_first) {
                t_first = t[1] - t_start;
            }
            t_last = t[1] - t_start;
        }
#endif
        for (uint32_t i = 0; i < 7; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        for (uint32_t i = 0; i < 6; ++i) {
            if (dib[i]) {
                tb[i] += u[i + 1] - u[i];
                any = true;
            }
        }
        if (!any && w_n >= nexp_g && w_dn >= nexp_g && freed >= nexp_g && slots_done() && b_s >= total_rows &&
            x_cred >= total_rows && (!OWN || (l_q >= total_rows && l_cons >= total_rows && a_q >= total_rows)) &&
            pq >= total_rows && h_q >= total_rows && h_cons >= total_rows && (J != 0 || z_sent >= total_rows) &&
            drained >= total_rows && (GATHER0 || (g_n >= nexp_g && g_state == 0)) && xq >= total_rows && qo_done()) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#if defined(WL5_Y_RM)
    noc_async_posted_writes_flushed();
#if defined(WL5_YC)
    if (J == 0) {  // (the stream ends with the collectors: each fenced its row writes, then + 1 here)
        do {
            invalidate_l1_cache();
        } while (*W(WL5_YC_CRED + 3 * 16) < WL5_YC);
    }
#elif defined(WL5_Y_FENCE)
    // the posted row writes' fence: an acknowledged 4-B write into every DRAM bank (the result buffer's spare rows from
    // WL5_Y_FENCE, nobody reads them) after them on the same NoC and VC; its acks mean every row landed
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(dst_l1, os.get_noc_addr(WL5_Y_FENCE + i), 4);
    }
    noc_async_write_barrier();
#endif
#endif
    // (every connection's done signal before any master's wait: with one-way lanes two groups' masters each count the
    // other group's clients)
    for (uint32_t d = 0; d < NMUX; ++d) {
        tt::tt_fabric::fabric_client_disconnect(mux[d]);
        if (!master[d]) {
            noc_semaphore_inc(get_noc_addr(tmx[d], tmy[d], term_sync[d]), 1);
            noc_async_atomic_barrier();
        }
    }
    for (uint32_t d = 0; d < NMUX; ++d) {
        if (master[d]) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(term_sync[d]), mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(muxx[d], muxy[d], mux_term);
        }
    }
    st[58] = n_fwd;
    st[59] = n_credit;
    for (uint32_t i = 0; i < 7; ++i) {
        st[470 + i] = tq[i];
    }
#if defined(WL5_HMC2)
    for (uint32_t i = 0; i < 6; ++i) {  // (the BRISC has its own: this RISC's x / gather / h out / own at 64-69)
        st[64 + i] = tb[i];
    }
#else
    for (uint32_t i = 0; i < 6; ++i) {  // (the queues l5_writer.cpp's BRISC runs on the other path, its stamp words)
        st[50 + i] = tb[i];
    }
#endif
    st[56] = n_own;
#if defined(WL5_WHY)
    for (uint32_t i = 0; i < 5; ++i) {  // (idle iterations by blocker: nothing / arrival / credit / mux / refill)
        st[477 + i] = twhy[i];
    }
    st[482] = t_first;
    st[483] = t_last;
#endif
    st[484] = wall() - t_start;
    st[485] = t_start;
#if !defined(WL5_HMC2)
    if (J != 0) {  // (no BRISC kernel on this core: its loop-start words as this RISC's)
        st[495] = st[484];
        st[496] = t_start;
    }
#endif
}
