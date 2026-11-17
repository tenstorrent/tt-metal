// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design with 5 x 5 K-split groups (ws_lane5.py), group NCRISC (NOC1). Core (C, Y) of a group,
// j = 5 C + Y. Weights as wl_reader.cpp (K typed slots in the chip's static order, forwarded packet by packet to the
// same core of the next chip through the group's mux of each direction; credits one hop back), the exchange as
// g5_reader.cpp, with the h rows as a unicast fan-out (no multicasts). A non-blocking loop over:
//   QW weights: compute order entry n (expert, slot s): once slot s's current occupant is that expert, push its
//      gate/up (CB 1) when its first GU_PK packets arrived and its down (CB 3) when all PK have.
//   QF forward: a slot's occupant that goes on in direction d sends packet u once it arrived, the first one only once
//      the next chip's slot freed its previous occupant (credit); fused write + arrival increment (no ack wait).
//   QR release: a slot frees once compute popped its expert and every forward was sent; the next occupant's sender
//      is told (own expert: FREEW for the BRISC; else the previous chip's core, inline write through the other mux).
//   QB x ring, QL landing (owners), QU partials to the owners above: as g5_reader.cpp.
//   QHF h stage B: once the 4 tiles of this core's column for row q landed (stage A, the owners' BRISCs), forward them
//      as one block to the 4 other cores of row Y (same slot and index, + column flag (slot, C) = q + 1), once each
//      consumed row q - NS (HCRED words 5 + c).
//   QH h ring: push row q (CB 2, 20 tiles) once its own column's 4 producer flags and the 4 other columns' block flags
//      of slot q % NS read >= q + 1; when compute pops h rows, the count into HCRED word Y at the column's owners and
//      word 5 + C at the row's other cores (the writers of this core's slots).
//   QY down outputs: every expert's rows, packed in the chip's order from tile row g ORMAX (DCOLS columns from DS).
// Slot tables: as wl_reader.cpp (loaded by the BRISC). Weight ring: one L1 tensor, gate/up slots [K][WGU] then down
// slots [K][WDN] (WGU = KSL 8, WDN = KTD DCOLS; owners 200 / 80, row 4 160 / 160 tiles).
// CT: 0 KSL, 1 DCOLS, 2 OWN, 3 NXR, 4 xflags, 5 xcred, 6 D, 7 land, 8 lflags, 9 lcred, 10 p tile, 11 scratch,
//     12 KTD, 13 NS, 14 hflags, 15 counts L1, 16 table L1, 17 warr, 18 ORMAX, 19 HT, 20 weight ring, 21 cred,
//     22 FREEW, 23 CHT, 24 W tile bytes, 25 PD, 26 partial ring, 27 HCRED, 28 K, 29 column block flags (NS x 5),
//     30 h ring, 31 h tile, 32-36 mux client CT args (buffers, buffer bytes, status, termination signal, clients), then
//     TensorAccessorArgs(out)
// RT: 0 j, 1 g, 2 order length, 3 stamps, 4 out addr, 5 DS, 6 mesh id, 7 next chip d0, 8 next chip d1, 9 this core's
//     xy, 10-14 column NOC x, 15-19 row NOC y, then the mux client RT args (17 words) of direction 0 and 1
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
    constexpr uint32_t NR = 5, NCOL = 5, NOWN = 4, PC = 4, NW = 8, LT = 2 * NR, NH = NOWN * NCOL;
    constexpr uint32_t WGU = KSL * NW, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;
    static_assert(WGU % CHT == 0 && WDN % CHT == 0);
    static_assert(K <= 7);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));
    const uint32_t DS = get_arg_val<uint32_t>(5);
    const uint16_t my_mesh = get_arg_val<uint32_t>(6);
    const uint8_t nchip[2] = {(uint8_t)get_arg_val<uint32_t>(7), (uint8_t)get_arg_val<uint32_t>(8)};
    const uint32_t myxy = get_arg_val<uint32_t>(9);
    const uint32_t mex = myxy & 0xFF, mey = myxy >> 8;
    uint32_t cx[NCOL], ry[NR];
    for (uint32_t i = 0; i < NCOL; ++i) {
        cx[i] = get_arg_val<uint32_t>(10 + i);
    }
    for (uint32_t i = 0; i < NR; ++i) {
        ry[i] = get_arg_val<uint32_t>(15 + i);
    }
    const uint32_t C = J / NR, Y = J % NR;
    auto W = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };

    volatile tt_l1_ptr uint32_t* counts = W(counts_l1);
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    volatile tt_l1_ptr uint32_t* tab = W(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

    // ---- mux connections (index d: to the next chip in direction d), headers
    tt::tt_fabric::WorkerToFabricMuxSender<mux_nbuf> mux[2];
    bool master[2];
    uint32_t term_sync[2], tmx[2], tmy[2];
    uint8_t muxx[2], muxy[2];
    for (uint32_t d = 0; d < 2; ++d) {
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
        mux[d] = tt::tt_fabric::build_connection_to_fabric_endpoint<mux_nbuf>(
            muxx[d],
            muxy[d],
            chan,
            mux_nbuf,
            mux_bufsz,
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
    auto qf = [&]() -> bool {
        for (uint32_t t = 0; t < 2 * K; ++t) {
            const uint32_t s = (rr + t) % K, d = ((rr + t) / K) % 2;
            if (!((cur_fwd[s] >> d) & 1) || sent[s][d] >= PK) {
                continue;
            }
            const uint32_t u = sent[s][d];
            if (warr[s * 4] < base[s] + u + 1) {
                continue;
            }
            const uint32_t s2 = cur_dst[s][d] & 7, k2 = cur_dst[s][d] >> 3;
            if (u == 0 && cred[(d * K + s2) * 4] < k2) {
                continue;
            }
            if (!mux[d].edm_has_space_for_packet()) {
                continue;
            }
            noc_async_writes_flushed();  // (the previous send's header and payload out of L1)
            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &mux[d],
                hdr[d],
                nchip[d],
                my_mesh,
                slot_addr(s, u),
                CHT * wtile,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(mex, mey, slot_addr(s2, u)), get_noc_addr(mex, mey, warr_addr + s2 * 16), 1, false});
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
            if (nk >= 2 && !mux[1 - (nk - 2)].edm_has_space_for_packet()) {
                continue;
            }
            noc_async_writes_flushed();  // (every forward of the slot has left L1 before it is refilled)
            if (nk == 1) {
                freew[s * 4] = next;
            } else if (nk >= 2) {
                const uint32_t d = nk - 2;
                fabric_unicast_noc_unicast_inline_write(
                    &mux[1 - d],
                    hcr,
                    nchip[1 - d],
                    my_mesh,
                    tt::tt_fabric::NocUnicastInlineWriteCommandHeader{
                        get_noc_addr(mex, mey, cred_addr + (d * K + s) * 16), next});
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
            for (uint32_t o = 0; o < Y; ++o) {
                noc_inline_dw_write(get_noc_addr(cx[C], ry[o], lcred + Y * 16), l_cons);
            }
            did = true;
        }
        return did;
    };

    // ---- QU: partials to the owners above
    const uint32_t nup = Y < NOWN ? Y : NOWN;
    uint32_t uq = 0, uo = 0;
    auto qu = [&]() -> bool {
        if (uq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*W(scr + 4 * 16) < uq + 1) {  // PAVAIL
            return false;
        }
        if (uo == 0) {
            *W(scr + 3 * 16) = uq + 1;
        }
        const uint32_t src = pring + (uq % PD) * NW * ptile, s = uq % D;
        bool did = false;
        while (uo < nup) {
            if (uq >= D && *W(lcred + uo * 16) < uq - D + 1) {
                return did;
            }
            noc_async_write(
                src + 2 * uo * ptile, get_noc_addr(cx[C], ry[uo], land + (s * LT + 2 * Y) * ptile), 2 * ptile);
            noc_async_write(scr + 3 * 16, get_noc_addr(cx[C], ry[uo], lflags + (s * NR + Y) * 16), 4);
            ++uo;
            did = true;
        }
        noc_async_writes_flushed();
        *W(scr + 5 * 16) = uq + 1;  // PDONE
        ++uq;
        uo = 0;
        return true;
    };

    // ---- QHF / QH: h stage B forwards, h ring rows, consumed counts to this core's slot writers
    auto col_in = [&](uint32_t q) {  // the 4 tiles of this core's column for row q landed (stage A)
        for (uint32_t o = 0; o < NOWN; ++o) {
            if (*W(hflags + ((q % NS) * NH + PC * C + o) * 16) < q + 1) {
                return false;
            }
        }
        return true;
    };
    uint32_t f_q = 0;
    auto qhf = [&]() -> bool {
        if (f_q >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (!col_in(f_q)) {
            return false;
        }
        if (f_q >= NS) {
            for (uint32_t t = 1; t < NCOL; ++t) {
                if (*W(hcred + (5 + (C + t) % NCOL) * 16) < f_q - NS + 1) {
                    return false;
                }
            }
        }
        const uint32_t blk = hring + ((f_q % NS) * KTD + PC * C) * htile, fl = hcolf + ((f_q % NS) * NCOL + C) * 16;
        *W(scr + 7 * 16) = f_q + 1;
        for (uint32_t t = 1; t < NCOL; ++t) {
            const uint32_t tc = (C + NCOL - t) % NCOL;  // (westwards first: NOC1's direction)
            noc_async_write(blk, get_noc_addr(cx[tc], ry[Y], blk), PC * htile);
            noc_async_write(scr + 7 * 16, get_noc_addr(cx[tc], ry[Y], fl), 4);
        }
        noc_async_writes_flushed();
        ++f_q;
        return true;
    };
    uint32_t h_q = 0, h_pages = 0, h_cons = 0;
    const uintptr_t hacked = (uintptr_t)get_cb_tiles_acked_ptr(hr_cb);
    uint16_t h_last = (uint16_t)reg_read(hacked);
    auto qh = [&]() -> bool {
        bool did = false;
        if (h_q < total_rows && h_q < f_q && cb_pages_reservable_at_back(hr_cb, KTD)) {
            invalidate_l1_cache();
            bool all = true;
            for (uint32_t t = 1; t < NCOL && all; ++t) {
                all = *W(hcolf + ((h_q % NS) * NCOL + (C + t) % NCOL) * 16) >= h_q + 1;
            }
            if (all) {  // (own column: col_in(h_q) held when it was forwarded, h_q < f_q)
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
            for (uint32_t o = 0; o < NOWN; ++o) {  // (stage A writers: the column's owners)
                noc_inline_dw_write(get_noc_addr(cx[C], ry[o], hcred + Y * 16), h_cons);
            }
            for (uint32_t t = 1; t < NCOL; ++t) {  // (stage B writers: the row's other cores)
                noc_inline_dw_write(get_noc_addr(cx[(C + t) % NCOL], ry[Y], hcred + (5 + C) * 16), h_cons);
            }
            did = true;
        }
        return did;
    };

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

    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    const uint32_t t_start = wall();
    uint32_t tq[7] = {0, 0, 0, 0, 0, 0, 0};
    while (true) {
        bool did[7];
        uint32_t t[8];
        t[0] = wall();
        invalidate_l1_cache();
        did[0] = qw() | qf() | qr();
        t[1] = wall();
        did[1] = qb();
        t[2] = wall();
        did[2] = ql();
        t[3] = wall();
        did[3] = qu();
        t[4] = wall();
        did[4] = qh();
        t[5] = wall();
        did[5] = qy();
        t[6] = wall();
        did[6] = qhf();
        t[7] = wall();
        bool any = false;
        for (uint32_t i = 0; i < 7; ++i) {
            if (did[i]) {
                tq[i] += t[i + 1] - t[i];
                any = true;
            }
        }
        if (!any && w_n >= nexp_g && w_dn >= nexp_g && freed >= nexp_g && slots_done() && b_s >= total_rows &&
            x_cred >= total_rows && (!OWN || (l_q >= total_rows && l_cons >= total_rows)) && uq >= total_rows &&
            h_q >= total_rows && h_cons >= total_rows && f_q >= total_rows && drained >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    for (uint32_t d = 0; d < 2; ++d) {
        tt::tt_fabric::fabric_client_disconnect(mux[d]);
        if (master[d]) {
            noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(term_sync[d]), mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(muxx[d], muxy[d], mux_term);
        } else {
            noc_semaphore_inc(get_noc_addr(tmx[d], tmy[d], term_sync[d]), 1);
            noc_async_atomic_barrier();
        }
    }
    st[58] = n_fwd;
    st[59] = n_credit;
    for (uint32_t i = 0; i < 7; ++i) {
        st[470 + i] = tq[i];
    }
    st[484] = wall() - t_start;
    st[485] = t_start;
}
