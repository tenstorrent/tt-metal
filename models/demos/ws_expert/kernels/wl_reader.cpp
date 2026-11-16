// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design, group NCRISC (NOC1): the group reader of ws_group (wg_reader.cpp) with the weights in K slots
// in this chip's static order (ws_lane_sched) and forwarded by this core itself, packet by packet, to the same core of
// the next chip through the group's mux core of each direction. A non-blocking loop over:
//   QW weights: compute order entry n (expert, slot s): once slot s's current occupant is that expert, push its gate/up
//      to compute (CB 1) when its first GU_PK packets have arrived and its down (CB 3) when all PK have.
//   QF forward: the occupant of each slot that goes on in direction d sends packet u once it has arrived (the slot's
//      arrival count, fabric atomic incs, reaches PK k + u + 1), the first one only once the next chip's slot s' has
//      freed its previous occupant (credit word d K + s' >= k'); a fused write + atomic inc of its arrival word.
//   QR release: a slot frees once compute popped its expert and every forward was sent; the next occupant's sender is
//      told: the BRISC through the local word FREEW + 16 s (own expert, read from DRAM), or the core of the chip it
//      comes from through a fabric inline write of k + 1 into its credit word (through the other direction's mux).
//   QA / QB / QH / QL / QY: x multicasts (column 1), x ring, h ring, h leader multicasts (column 1), down outputs.
// Every wait of QW / QF / QR / the BRISC's own reads is on an event earlier in the schedule's model time, and the
// fabric never blocks for long (every packet's landing slot is free before it is sent), so the lanes cannot deadlock.
// Slot tables (L1, loaded by the BRISC): word 0 order length, 1..K occupants a slot, K+1..2K slot table offsets,
// order from word 16 (e | s << 16); occupant k of slot s: 4 words at offset_s + 4 k: e | kind << 16 | fwd << 18 |
// next kind << 20 (kind: 1 own, 2 from direction 0, 3 from direction 1), s'0 | k'0 << 2, s'1 | k'1 << 2, own index.
// RT: 0 j, 1 g, 2 order length, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 stamps, 7 out addr, 8 mesh id,
//     9 next chip direction 0, 10 next chip direction 1, 11 this core's xy, then the mux client RT args (17 words)
//     of direction 0 and of direction 1
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 GC, 5 G, 6 NS, 7 NX, 8 hflags, 9 xring, 10 xflags, 11 xprog, 12-15 group rect
//     for NOC1, 16 x tile bytes, 17 counts L1, 18 table L1, 19 hring, 20 h tile bytes, 21 warr (arrivals a slot,
//     16 B apart), 22 ORMAX (output tile rows a group), 23 HT, 24 wgu ring, 25 wd ring, 26 cred (2K words), 27 FREEW
//     (K words), 28 CHT, 29 W tile bytes, 30-34 mux client CT args (buffers, buffer bytes, status, termination
//     signal, clients), then TensorAccessorArgs(out)
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
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t SEM_HARR = 1, SEM_SCR2 = 3, SEM_SCRH = 5;
#ifndef WL_WPOLL
#define WL_WPOLL 1
#endif
#ifndef WL_K
#define WL_K 3
#endif
constexpr uint32_t K = WL_K;  // weight slots a core
static_assert(K >= 1 && K <= 4);  // (2-bit slot fields in the tables)

// A group multicast and its flag through the same command buffer, the flag as a 4-byte multicast write (not
// noc_semaphore_set_multicast): on this architecture the write and write-register command buffers are not sent in
// issue order, so a flag from the other buffer could overtake its payload; from one buffer, on one NoC and virtual
// channel, the flag lands after its payload. The payload is unlinked: a linked multicast keeps its path reserved
// until the source's next write, and several sources multicast into a group's rectangle on each NoC
// (WL_MCAST_LINKED links it, the flag then closing the path). Callers flush after each pair, so a core has at most
// one multicast in its NoC's queue.
FORCE_INLINE void mcast_with_flag(
    uint32_t src, uint64_t dst, uint32_t bytes, uint32_t n, uint32_t flag_src, uint64_t flag_dst) {
#ifdef WL_MCAST_LINKED
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, true);
#else
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, false);
#endif
    noc_async_write_multicast_loopback_src(flag_src, flag_dst, 4, n, false);
}

void kernel_main() {
    constexpr uint32_t KT = get_compile_time_arg_val(0);
    constexpr uint32_t BK = get_compile_time_arg_val(1);
    constexpr uint32_t KTD = get_compile_time_arg_val(2);
    constexpr uint32_t DCOLS = get_compile_time_arg_val(3);
    constexpr uint32_t GC = get_compile_time_arg_val(4);
    [[maybe_unused]] constexpr uint32_t G = get_compile_time_arg_val(5);
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
    constexpr uint32_t table_l1 = get_compile_time_arg_val(18);
    constexpr uint32_t hring = get_compile_time_arg_val(19), htile = get_compile_time_arg_val(20);
    constexpr uint32_t warr_addr = get_compile_time_arg_val(21);
    constexpr uint32_t ORMAX = get_compile_time_arg_val(22), HT = get_compile_time_arg_val(23);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(24), wd_ring = get_compile_time_arg_val(25);
    constexpr uint32_t cred_addr = get_compile_time_arg_val(26), freew_addr = get_compile_time_arg_val(27);
    constexpr uint32_t CHT = get_compile_time_arg_val(28), wtile = get_compile_time_arg_val(29);
    constexpr uint8_t mux_nbuf = get_compile_time_arg_val(30);
    constexpr size_t mux_bufsz = get_compile_time_arg_val(31);
    constexpr size_t mux_status = get_compile_time_arg_val(32);
    constexpr size_t mux_term = get_compile_time_arg_val(33);
    constexpr uint32_t mux_clients = get_compile_time_arg_val(34);
    constexpr auto oargs = TensorAccessorArgs<35>();
    constexpr uint32_t BPR = KT / BK, HALF = GC / 2;
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;  // packets of a slice (gate / up first)
    static_assert(WGU % CHT == 0 && WDN % CHT == 0);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint16_t my_mesh = get_arg_val<uint32_t>(8);
    const uint8_t nchip[2] = {(uint8_t)get_arg_val<uint32_t>(9), (uint8_t)get_arg_val<uint32_t>(10)};
    const uint32_t myxy = get_arg_val<uint32_t>(11);
    const uint32_t mex = myxy & 0xFF, mey = myxy >> 8;

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    do {
        invalidate_l1_cache();
    } while (counts[0] == 0);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
    uint32_t total_rows = 0, total_x = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }
    total_x = total_rows * BPR;

    // ---- mux connections (index d: to the next chip in direction d), headers
    tt::tt_fabric::WorkerToFabricMuxSender<mux_nbuf> mux[2];
    bool master[2];
    uint32_t term_sync[2], tmx[2], tmy[2];
    uint8_t muxx[2], muxy[2];
    for (uint32_t d = 0; d < 2; ++d) {
        size_t a_ = 12 + 17 * d;
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
            muxx[d], muxy[d], chan, mux_nbuf, mux_bufsz, ch_base, conn_info, handshake, flow, bufidx, l_flow,
            l_teardown, l_bufidx);
        tt::tt_fabric::wait_for_fabric_endpoint_ready(muxx[d], muxy[d], mux_status, l_status);
        tt::tt_fabric::fabric_client_connect(mux[d]);
    }
    PacketHeaderPool::reset();
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hcr = PacketHeaderPool::allocate_header();

    // ---- slots: current occupant (its table entry cached in registers), its packets sent a direction, computed
    volatile tt_l1_ptr uint32_t* warr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(warr_addr);
    volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred_addr);
    volatile tt_l1_ptr uint32_t* freew = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freew_addr);
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
        return u < GU_PK ? wgu_ring + (s * WGU + u * CHT) * wtile : wd_ring + (s * WDN + (u - GU_PK) * CHT) * wtile;
    };

    // ---- QW: push weights to compute in the chip's order; track what compute popped
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
        while (freed < f) {  // order entry `freed` is computed: its slot's current occupant
            done_c[order[freed] >> 16] = true;
            ++freed;
            did = true;
        }
        return did;
    };

    // ---- QF: forward the slots' occupants, a packet a call; QR: release finished slots
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
            const uint32_t s2 = cur_dst[s][d] & 3, k2 = cur_dst[s][d] >> 2;
            if (u == 0 && cred[(d * K + s2) * 4] < k2) {
                continue;
            }
            if (!mux[d].edm_has_space_for_packet()) {  // (never block this RISC: its x / h queues feed the group)
                continue;
            }
            // (the previous send's header and payload must have left L1 before the header is reused: flushed here
            // rather than right after each send, so the payload drains while this RISC serves its other queues)
            noc_async_writes_flushed();
            fabric_unicast_noc_fused_unicast_with_atomic_inc(
                &mux[d],
                hdr[d],
                nchip[d],
                my_mesh,
                slot_addr(s, u),
                CHT * wtile,
                tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                    get_noc_addr(mex, mey, slot_addr(s2, u)), get_noc_addr(mex, mey, warr_addr + s2 * 16), 1, true});
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
                freew[s * 4] = next;  // (the BRISC reads the chip's own next expert into it)
            } else if (nk >= 2) {
                // next occupant arrives on direction d = nk - 2 from the previous chip in d: through mux 1 - d
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

    // x stream helpers: block s -> (expert, index in it); walked forward only
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    uint32_t t_cred = 0, t_empty = 0, n_cred = 0, last_t = wall();
    // ---- QA: own x multicasts
    const uint32_t L0 = 2 * (J % HALF) + J / HALF;
    uint32_t a_n = 0, a_l = L0, a_base = 0;
    const uint32_t scr_addr = get_semaphore(SEM_SCR2);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    auto qa = [&]() -> bool {
        while (a_n < nexp_g && a_l >= rows_of(a_n) * BPR) {
            a_base += rows_of(a_n) * BPR;
            ++a_n;
            a_l = L0;
        }
        const uint32_t now = wall();
        const uint32_t dt = now - last_t;
        last_t = now;
        if (a_n >= nexp_g || J / HALF != 1) {  // (column 0 multicasts x from its BRISC, on NOC0)
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
#if WG_XORDER
        if (s >= WG_XORDER) {  // in stream order: block s - WG_XORDER has landed here (one multicast at a time a NoC)
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + ((s - WG_XORDER) % NX) * 16) <
                s - WG_XORDER + 1) {
                return false;
            }
        }
#endif
        const uint32_t xs = s % NX;
        *scr = s + 1;
        mcast_with_flag(
            get_read_ptr(xst_cb),
            get_noc_multicast_addr(mx0, my0, mx1, my1, xring + xs * BK * xtile),
            BK * xtile,
            GC,
            scr_addr,
            get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + xs * 16));
        // (flushed, not barrier: only the source must be free before reuse; a barrier waited for 20 acks and stalled
        // every other queue of this RISC)
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, BK);
#ifdef WG_TRACE0
        if (s >= WG_TRACE0 && s < WG_TRACE0 + 40) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6))[128 + s - WG_TRACE0] = wall();
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
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6))[192 + b_s - WG_TRACE0] =
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
                const uint32_t ol = nxt - w_base, owner = (ol % 2) * HALF + (ol / 2) % HALF;
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
    uint32_t mx[4] = {0, 0, 0, 0}, tot[4] = {0, 0, 0, 0};
    // ---- QL: column 1's h leader multicasts (rows q with q % 10 == Y; see the BRISC's Q3)
    const uint32_t C = J / HALF, Y = J % HALF;
    uint32_t led = 0, lq = Y;
    const uint32_t harr = get_semaphore(SEM_HARR), hscr_addr = get_semaphore(SEM_SCRH);
    volatile tt_l1_ptr uint32_t* hscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hscr_addr);
    auto ql = [&]() -> bool {
        if (C != 1 || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(harr) < HALF * (led + 1)) {
            return false;
        }
        const uint32_t part = hring + ((lq % NS) * KTD + C * HALF) * htile;
        *hscr = lq + 1;
        mcast_with_flag(
            part,
            get_noc_multicast_addr(mx0, my0, mx1, my1, part),
            HALF * htile,
            GC,
            hscr_addr,
            get_noc_multicast_addr(mx0, my0, mx1, my1, hflags + ((lq % NS) * 2 + C) * 16));
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };
#ifdef WG_Y_NCRISC
    // ---- QY: down outputs (CB 17) to DRAM, on NOC1 (the BRISC's NOC0 keeps the DRAM reads): the group's last
    // expert's rows at tile row g ORMAX, or (WL_OUT_ALL) every expert's, packed in order from g ORMAX
    constexpr uint32_t y_cb = tt::CBIndex::c_17;
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(7), obytes);
    uint32_t drained = 0, d_n = 0, d_r = 0, d_base = 0;
    auto qy = [&]() -> bool {
        if (drained >= total_rows || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        while (d_r >= rows_of(d_n)) {
            d_base += rows_of(d_n);
            ++d_n;
            d_r = 0;
        }
#ifdef WL_OUT_ALL
        if (true) {
            const uint32_t row = g * ORMAX + d_base + d_r;
#else
        if (d_n + 1 == nexp_g) {
            const uint32_t row = g * ORMAX + d_r;
#endif
            uint32_t l1 = get_read_ptr(y_cb);
            for (uint32_t t = 0; t < DCOLS; ++t, l1 += obytes) {
                noc_async_write_page(row * HT + J * DCOLS + t, os, l1);
            }
            noc_async_write_barrier();
        }
        cb_pop_front(y_cb, DCOLS);
        ++d_r;
        ++drained;
        return true;
    };
#else
    const uint32_t drained = total_rows;
    auto qy = []() -> bool { return false; };
#endif
    uint32_t iter = 0;
    while (true) {
        const uint32_t t0 = wall();
        const bool y = qy();
        const bool l = ql();
        bool a = false;
        if (++iter % WL_WPOLL == 0) {  // (the weight queues every WL_WPOLL turns: the x / h queues pace compute)
            invalidate_l1_cache();
            a = qw() | qf() | qr();
        }
        const uint32_t t1 = wall();
        const bool b = qa();
        const uint32_t t2 = wall();
        const bool c = qb();
        const uint32_t t3 = wall();
        const bool d = qh();
        const uint32_t t4 = wall();
        const uint32_t dd[4] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3};
        for (uint32_t i = 0; i < 4; ++i) {
            tot[i] += dd[i];
            mx[i] = dd[i] > mx[i] ? dd[i] : mx[i];
        }
        if (!y && !l && !a && !b && !c && !d && drained >= total_rows && (C != 1 || lq >= total_rows) &&
            w_n >= nexp_g && w_dn >= nexp_g && freed >= nexp_g && slots_done() && (a_n >= nexp_g || J / HALF != 1) &&
            b_s >= total_x && x_credited >= total_x && h_q >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    // mux teardown: every client disconnects; client 0 of each mux stops it once all others have
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
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    st[48] = t_cred;
    st[49] = t_empty;
    st[58] = n_fwd;
    st[59] = n_credit;
    for (uint32_t i = 0; i < 4; ++i) {
        st[470 + i] = tot[i];
        st[474 + i] = mx[i];
    }
}
