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
// next kind << 20 (kind: 1 own, 2 from direction 0, 3 from direction 1), s'0 | k'0 << 3, s'1 | k'1 << 3, own index.
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

// WL_HCRED: h ring slot credits (see wl_writer.cpp): QH publishes this core's consumed h rows
#ifdef WL_HCRED
#define HCRED_OK(word, q) \
    ((q) < NS || *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HCRED + (word) * 16) >= (q) - NS + 1)
#else
#define HCRED_OK(word, q) true
#endif
#ifdef WL_HA2A
#include "wl_ha2a.hpp"
#endif
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
// WL_MC_NOC1: column 0's x and h-leader multicasts also go out from its NCRISC on NOC1 (none from the BRISCs on NOC0)
// WL_BCAST: no column multicasts here; the x blocks of cores 1.. go on NOC1 into core 0's funnel slots (QA), the ring
// credits all go to core 0, and core GC - 1 multicasts every h row on NOC1 (QL); see wl_writer.cpp
#if defined(WL_BCAST)
constexpr uint32_t MC_COL_MASK = 0;  // (columns that multicast from this RISC: bit c)
#elif defined(WL_MC_NOC1)
constexpr uint32_t MC_COL_MASK = 3;
#else
constexpr uint32_t MC_COL_MASK = 2;
#endif
constexpr uint32_t K = WL_K;  // weight slots a core
static_assert(K >= 1 && K <= 7);  // (3-bit slot fields in the tables)

// A group multicast and its flag through the same command buffer, the flag as a 4-byte multicast write (not
// noc_semaphore_set_multicast): on this architecture the write and write-register command buffers are not sent in
// issue order, so a flag from the other buffer could overtake its payload; from one buffer, on one NoC and virtual
// channel, the flag lands after its payload. The payload is unlinked: a linked multicast keeps its path reserved
// until the source's next write, and several sources multicast into a group's rectangle on each NoC
// (WL_MCAST_LINKED links it, the flag then closing the path). Callers flush after each pair, so a core has at most
// one multicast in its NoC's queue.
FORCE_INLINE void mcast_with_flag(
    uint32_t src, uint64_t dst, uint32_t bytes, uint32_t n, uint32_t flag_src, uint64_t flag_dst) {
#if defined(WL_MC_FLUSHPKT)
    // (diagnostic) one packet at a time, each flushed (sent) before the command buffer is programmed again: no
    // multicast still waiting for its path reservation can be overwritten by the next packet or the flag
    for (uint32_t o = 0; o < bytes; o += NOC_MAX_BURST_SIZE) {
        const uint32_t m = bytes - o < NOC_MAX_BURST_SIZE ? bytes - o : NOC_MAX_BURST_SIZE;
        noc_async_write_multicast_loopback_src(src + o, dst + o, m, n, false);
        noc_async_writes_flushed();
    }
    noc_async_write_multicast_loopback_src(flag_src, flag_dst, 4, n, false);
    noc_async_writes_flushed();
#else
#ifdef WL_MCAST_LINKED
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, true);
#else
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, false);
#endif
    noc_async_write_multicast_loopback_src(flag_src, flag_dst, 4, n, false);
#endif
}

// WL_UCAST (the default, LCfg.ucast): no NoC multicasts at all. A group "multicast" is a unicast fan-out to the
// group's 2 x HALF cores (columns cx[0], cx[1], rows ny0 .. ny0 + HALF - 1), this core included: the payload to every
// core, then the 4-byte flag to every core, all from the write command buffer on this RISC's NoC, so each core's flag
// lands after its own payload (one path, one virtual channel). A payload whose source is its own destination (the h
// leader's row part) is not copied onto itself (core me = c * HALF + y), only flagged.
template <uint32_t HALF>
FORCE_INLINE void group_write_with_flag(
    const uint32_t* cx,
    uint32_t ny0,
    uint32_t me,
    uint32_t src,
    uint32_t dst,
    uint32_t bytes,
    uint32_t flag_src,
    uint32_t flag_dst) {
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t y = 0; y < HALF; ++y) {
            if (src != dst || c * HALF + y != me) {
                noc_async_write(src, get_noc_addr(cx[c], ny0 + y, dst), bytes);
            }
        }
    }
    for (uint32_t c = 0; c < 2; ++c) {
        for (uint32_t y = 0; y < HALF; ++y) {
            noc_async_write(flag_src, get_noc_addr(cx[c], ny0 + y, flag_dst), 4);
        }
    }
}

// WL_XCHAIN: see wl_writer.cpp (this RISC: column 1's chain starts, west on NOC1, and the hops one row up)
FORCE_INLINE void x_hop(
    uint32_t src, uint32_t x, uint32_t y, uint32_t dst, uint32_t bytes, uint32_t flag_src, uint32_t flag_dst) {
    noc_async_write(src, get_noc_addr(x, y, dst), bytes);
    noc_async_write(flag_src, get_noc_addr(x, y, flag_dst), 4);
}

// WL_XP / WL_HP (chains): an x block / h row part (T tiles of tb bytes) moves in P pieces, split on tile boundaries,
// piece p followed by its own flag at word p of the slot's 16 B (same command buffer, NoC and VC: after its piece),
// sent from word p of the 16 B flag source (a NoC write keeps the address's offset within 16 B: an origin's scratch
// holds its value in all four words). A forwarder passes piece p on as soon as its flag has landed (cut-through), from
// its own flag word p (value s + 1, stable until the slot's credit); a slot is complete once word P - 1 has landed
// (every piece of a slot comes from one sender, in order).
#ifndef WL_XP
#define WL_XP 1
#endif
#ifndef WL_HP
#define WL_HP 1
#endif
constexpr uint32_t XP = WL_XP, HP = WL_HP, XL = (XP - 1) * 4, HL = (HP - 1) * 4;
template <uint32_t T, uint32_t P>
FORCE_INLINE void piece_data(uint32_t p, uint32_t src, uint32_t x, uint32_t y, uint32_t dst, uint32_t tb) {
    const uint32_t o = (p * T / P) * tb, n = ((p + 1) * T / P) * tb - o;
    noc_async_write(src + o, get_noc_addr(x, y, dst + o), n);
}
FORCE_INLINE void piece_flag(uint32_t p, uint32_t x, uint32_t y, uint32_t flag_src, uint32_t flag_dst) {
    noc_async_write(flag_src + 4 * p, get_noc_addr(x, y, flag_dst + 4 * p), 4);
}
template <uint32_t T, uint32_t P>
FORCE_INLINE void piece_hop(
    uint32_t p, uint32_t src, uint32_t x, uint32_t y, uint32_t dst, uint32_t tb, uint32_t flag_src, uint32_t flag_dst) {
    piece_data<T, P>(p, src, x, y, dst, tb);
    piece_flag(p, x, y, flag_src, flag_dst);
}

// WL_FASYNC: the forwarders issue a piece's payload and its flag only when this RISC's write command buffer is free
// (it stays busy while a packet's data goes out, ~55 B a cycle): else back to the loop, resuming at the same step, so
// the RISC does not stall in the issue for the NoC's injection time (busy forwarding was ~650K / 430K cycles of the
// BRISC / NCRISC at M 586)
FORCE_INLINE bool wr_ready() {
#ifdef WL_FASYNC
    return noc_cmd_buf_ready(noc_index, write_cmd_buf);
#else
    return true;
#endif
}

// Chain routing in a column of HALF rows for a block / h part whose origin is row r: does row y pass it one row down
// (BRISC, NOC0) / one row up (NCRISC, NOC1)? Plain chain: from r both ways. WL_JUMP: the origin also sends to row
// r' = (r + HALF / 2) % HALF of both columns, so each column has two entries lo < hi = lo + HALF / 2: lo feeds the rows
// above it and the next ND below it, hi the rows below it and the rest between (at most HALF / 2 - 1 hops, not HALF -
// 1).
template <uint32_t HALF>
FORCE_INLINE bool chain_down(uint32_t y, uint32_t r) {
#ifdef WL_JUMP
    static_assert(HALF % 2 == 0);
    constexpr uint32_t J2 = HALF / 2, ND = J2 / 2;
    const uint32_t lo = r < J2 ? r : r - J2, hi = lo + J2;
    return (y >= hi && y + 1 < HALF) || (y >= lo && y < lo + ND);
#else
    return r <= y && y + 1 < HALF;
#endif
}
template <uint32_t HALF>
FORCE_INLINE bool chain_up(uint32_t y, uint32_t r) {
#ifdef WL_JUMP
    constexpr uint32_t J2 = HALF / 2, ND = J2 / 2;
    const uint32_t lo = r < J2 ? r : r - J2, hi = lo + J2;
    return (y <= lo && y > 0) || (y >= lo + ND + 2 && y <= hi);
#else
    return r >= y && y > 0;
#endif
}

// WL_NOEX (diagnostic, wrong results): no x / h exchange at all - every ring is handed to compute as if it had
// arrived; the gather, tilize, h tiles and weight streaming still run

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
#ifdef WL_KS
    total_x = total_rows;  // (K split: the x ring holds whole rows)
    constexpr uint32_t KSL = WL_KS, KS_NW = 2 * HALF, KS_LT = 2 * HALF;
    auto KW = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
#else
    total_x = total_rows * BPR;
#endif

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
#ifdef WL_ARR_WRITE
    // arrival counts as plain writes: after each payload, a 16-byte write of the slot's absolute arrival count from a
    // staging word of this core (one a direction, at address WL_ARR_WRITE), through the same router command buffer
    // as the payload, so it lands after it; the routers then do no NoC atomics into the group cores
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hca[2] = {
        PacketHeaderPool::allocate_header(), PacketHeaderPool::allocate_header()};
    volatile tt_l1_ptr uint32_t* arr_stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_ARR_WRITE);
#endif

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
            const uint32_t s2 = cur_dst[s][d] & 7, k2 = cur_dst[s][d] >> 3;
            if (u == 0 && cred[(d * K + s2) * 4] < k2) {
                continue;
            }
#ifdef WL_ARR_WRITE
            if (!mux[d].template edm_has_space_for_packet<2>()) {  // (never block this RISC)
                continue;
            }
            noc_async_writes_flushed();  // (headers, payload and staging word of the previous send out of L1)
            fabric_unicast_noc_unicast_write(
                &mux[d],
                hdr[d],
                nchip[d],
                my_mesh,
                slot_addr(s, u),
                CHT * wtile,
                tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(mex, mey, slot_addr(s2, u))});
            arr_stage[d * 4] = PK * k2 + u + 1;
            fabric_unicast_noc_unicast_write(
                &mux[d],
                hca[d],
                nchip[d],
                my_mesh,
                WL_ARR_WRITE + d * 16,
                16,
                tt::tt_fabric::NocUnicastCommandHeader{get_noc_addr(mex, mey, warr_addr + s2 * 16)});
#else
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
                    get_noc_addr(mex, mey, slot_addr(s2, u)),
                    get_noc_addr(mex, mey, warr_addr + s2 * 16),
                    1,
#ifdef WL_ARR_NF
                    false});
#else
                    true});
#endif
#endif
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
    [[maybe_unused]] uint32_t t_cred = 0, t_empty = 0, n_cred = 0, last_t = wall();
    // ---- QA: own x multicasts
    const uint32_t L0 = 2 * (J % HALF) + J / HALF;
    uint32_t a_n = 0, a_l = L0, a_base = 0;
    const uint32_t scr_addr = get_semaphore(SEM_SCR2);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
#ifdef WL_BCAST
    // (cores 1..) own block s into core 0's funnel slot s % WL_XF once block s - WL_XF, the slot's previous one, has
    // landed here (so core 0 has read it out), then its filled flag through the same command buffer
    const bool mcx = J != 0;
    auto qa = [&]() -> bool {
        while (a_n < nexp_g && a_l >= rows_of(a_n) * BPR) {
            a_base += rows_of(a_n) * BPR;
            ++a_n;
            a_l = L0;
        }
        if (!mcx || a_n >= nexp_g || !cb_pages_available_at_front(xst_cb, BK)) {
            return false;
        }
        const uint32_t s = a_base + a_l;
        if (s >= WL_XF) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + ((s - WL_XF) % NX) * 16) < s - WL_XF + 1) {
                return false;
            }
        }
        *scr = s + 1;
        noc_async_write(
            get_read_ptr(xst_cb), get_noc_addr(cxs[0], ny0, WL_XIN + (s % WL_XF) * BK * xtile), BK * xtile);
        noc_async_write(scr_addr, get_noc_addr(cxs[0], ny0, WL_XINF + (s % WL_XF) * 16), 4);
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, BK);
        a_l += GC;
        return true;
    };
#elif defined(WL_KS)
    // ---- QA (K split, column 1): this core's tilized x rows (q % 2 == 1) -> its own and its partner's ring slot (NOC1
    // west), as the BRISC's QX
    const bool mcx = J / HALF == 1;
    uint32_t ka_q = 1;
    auto qa = [&]() -> bool {
#ifdef WL_KS_XDIRECT
        // (see the BRISC's QX: the row is in its ring slot here already)
        if (!mcx || ka_q >= total_rows || !cb_pages_available_at_front(xst_cb, 1)) {
            return false;
        }
        if (ka_q >= NX) {
            invalidate_l1_cache();
            if (*KW(WL_KS_XCRED + 16) < ka_q - NX + 1) {
                return false;
            }
        }
        {
            const uint32_t Cc = J / HALF, Yc = J % HALF;
            const uint32_t a = xring + (ka_q % NX) * KSL * xtile, fl = xflags + ((ka_q % NX) * 2 + Cc) * 16;
            *KW(WL_KS_SCR + 2 * 16) = ka_q + 1;
            noc_async_write(a, get_noc_addr(cxs[1 - Cc], ny0 + Yc, a), KSL * xtile);
            noc_async_write(WL_KS_SCR + 2 * 16, get_noc_addr(cxs[1 - Cc], ny0 + Yc, fl), 4);
            noc_async_writes_flushed();
            *KW(fl) = ka_q + 1;
            cb_pop_front(xst_cb, 1);
            ka_q += 2;
            return true;
        }
#endif
        if (!mcx || ka_q >= total_rows || !cb_pages_available_at_front(xst_cb, KSL)) {
            return false;
        }
        if (ka_q >= NX) {
            invalidate_l1_cache();
            if (*KW(WL_KS_XCRED) < ka_q - NX + 1 || *KW(WL_KS_XCRED + 16) < ka_q - NX + 1) {
                return false;
            }
        }
        const uint32_t Cc = J / HALF, Yc = J % HALF;
        const uint32_t dst = xring + (ka_q % NX) * KSL * xtile, fl = xflags + ((ka_q % NX) * 2 + Cc) * 16;
        *KW(WL_KS_SCR + 2 * 16) = ka_q + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        noc_async_write(src, get_noc_addr(cxs[Cc], ny0 + Yc, dst), KSL * xtile);
        noc_async_write(WL_KS_SCR + 2 * 16, get_noc_addr(cxs[Cc], ny0 + Yc, fl), 4);
        noc_async_write(src, get_noc_addr(cxs[1 - Cc], ny0 + Yc, dst), KSL * xtile);
        noc_async_write(WL_KS_SCR + 2 * 16, get_noc_addr(cxs[1 - Cc], ny0 + Yc, fl), 4);
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, KSL);
        ka_q += 2;
        return true;
    };
#else
    const bool mcx = (MC_COL_MASK >> (J / HALF)) & 1;
    auto qa = [&]() -> bool {
        while (a_n < nexp_g && a_l >= rows_of(a_n) * BPR) {
            a_base += rows_of(a_n) * BPR;
            ++a_n;
            a_l = L0;
        }
        const uint32_t now = wall();
        const uint32_t dt = now - last_t;
        last_t = now;
        if (a_n >= nexp_g || !((MC_COL_MASK >> (J / HALF)) & 1)) {  // (else column 0's BRISC multicasts on NOC0)
            return false;
        }
        if (!cb_pages_available_at_front(xst_cb, BK)) {
            t_empty += dt;
            return false;
        }
        const uint32_t s = a_base + a_l;
#ifdef WL_NOEX
        cb_pop_front(xst_cb, BK);
        a_l += GC;
        return true;
#endif
#ifdef WL_XHOP
        if (s >= NX) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 3 * 16) < s - NX + 1 ||
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 2 * 16) < s - NX + 1) {
                t_cred += dt;
                return false;
            }
        }
        if (false) {
#else
        if (s >= NX) {
#endif
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
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + ((s - WG_XORDER) % NX) * 16 + XL) <
                s - WG_XORDER + 1) {
                return false;
            }
        }
#endif
        const uint32_t xs = s % NX;
        for (uint32_t p = 0; p < 4; ++p) {  // (all four words: see piece_hop)
            scr[p] = s + 1;
        }
#if defined(WL_XCHAIN)
        {
            const uint32_t src = get_read_ptr(xst_cb), dst = xring + xs * BK * xtile, xf = xflags + xs * 16;
            for (uint32_t p = 0; p < XP; ++p) {
                piece_hop<BK, XP>(p, src, cxs[J / HALF], ny0 + J % HALF, dst, xtile, scr_addr, xf);
                piece_hop<BK, XP>(p, src, cxs[1 - J / HALF], ny0 + J % HALF, dst, xtile, scr_addr, xf);
#ifdef WL_JUMP
                for (uint32_t c2 = 0; c2 < 2; ++c2) {
                    piece_hop<BK, XP>(p, src, cxs[c2], ny0 + (J % HALF + HALF / 2) % HALF, dst, xtile, scr_addr, xf);
                }
#endif
            }
        }
#elif defined(WL_UCAST)
        group_write_with_flag<HALF>(
            cxs, ny0, J, get_read_ptr(xst_cb), xring + xs * BK * xtile, BK * xtile, scr_addr, xflags + xs * 16);
#else
    mcast_with_flag(
        get_read_ptr(xst_cb),
        get_noc_multicast_addr(mx0, my0, mx1, my1, xring + xs * BK * xtile),
        BK * xtile,
        GC,
        scr_addr,
        get_noc_multicast_addr(mx0, my0, mx1, my1, xflags + xs * 16));
#endif
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
#endif
    // ---- the chain forwarders (see the BRISC's QD / QHD: every landed piece of a block / part a call; WL_FNB: no
    // blocking flush, what was walked counts (credits, h rows to compute) once this RISC's writes have all left)
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* fwd_word =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
#ifdef WL_XCHAIN
    const uint32_t YR = J % HALF;
    uint32_t fu_s = YR > 0 ? 0 : total_x, fu_n = 0, fu_base = 0, fu_p = 0;
    bool fu_f = false;
#else
    const uint32_t fu_s = total_x;
#endif
#ifdef WL_HCHAIN
    const uint32_t total_hd = 2 * total_rows, YH = J % HALF, CH = J / HALF;
    uint32_t hu_i = YH > 0 ? 0 : total_hd, hu_p = 0;
    bool hu_f = false;
#else
    const uint32_t total_hd = 0, hu_i = 0;
#endif
    uint32_t fu_pub = fu_s, hu_pub = hu_i;
    [[maybe_unused]] auto fnb = [&]() -> bool {
#ifdef WL_FNB
        if ((fu_pub != fu_s || hu_pub != hu_i) && ncrisc_noc_nonposted_writes_sent(noc_index)) {
            fu_pub = fu_s;
            hu_pub = hu_i;
            return true;
        }
#endif
        return false;
    };
#ifdef WL_XCHAIN
    // ---- QU: x chain, one row up (NOC1): blocks in stream order, those whose origin row is >= this row
    auto qu = [&]() -> bool {
        bool did = fnb();
        if (fu_s >= total_x) {
            return did;
        }
        while (fu_s >= fu_base + rows_of(fu_n) * BPR) {
            fu_base += rows_of(fu_n) * BPR;
            ++fu_n;
        }
        if (chain_up<HALF>(YR, ((fu_s - fu_base) / 2) % HALF)) {
            const uint32_t xs = fu_s % NX, xf = xflags + xs * 16;
            do {
                if (!fu_f) {
                    invalidate_l1_cache();
                    if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xf + 4 * fu_p) < fu_s + 1) {
                        return did;
                    }
#ifdef WL_XHOP
                    if (fu_p == 0 && fu_s >= NX &&
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 1 * 16) <
                            fu_s - NX + 1) {  // (the row above has freed the slot)
                        return did;
                    }
#endif
                    if (!wr_ready()) {
                        return did;
                    }
                    piece_data<BK, XP>(
                        fu_p, xring + xs * BK * xtile, cxs[J / HALF], ny0 + YR - 1, xring + xs * BK * xtile, xtile);
                    fu_f = did = true;
                }
                if (!wr_ready()) {
                    return did;
                }
                piece_flag(fu_p, cxs[J / HALF], ny0 + YR - 1, xf, xf);
                fu_f = false;
            } while (++fu_p < XP);
            fu_p = 0;
#ifndef WL_FNB
            noc_async_writes_flushed();
#endif
        }
        ++fu_s;
#ifndef WL_FNB
        fu_pub = fu_s;
#endif
        return true;
    };
    // block s may be credited (its slot reused) once both forwarders have passed it (their hops left this core)
    auto fwd_past = [&](uint32_t s) { return fu_pub > s && (*fwd_word & 0xFFFF) > s; };
#else
    auto qu = []() -> bool { return false; };
    auto fwd_past = [](uint32_t) { return true; };
#endif
#ifdef WL_HCHAIN
    // ---- QHU: h chain, one row up (NOC1): row parts i = 2 q + c in order, those whose leader row is >= this row
    auto qhu = [&]() -> bool {
        bool did = fnb();
        if (hu_i >= total_hd) {
            return did;
        }
        const uint32_t q = hu_i / 2, c = hu_i % 2;
        if (chain_up<HALF>(YH, q % HALF)) {
            const uint32_t f = hflags + ((q % NS) * 2 + c) * 16;
            const uint32_t part = hring + ((q % NS) * KTD + c * HALF) * htile;
            do {
                if (!hu_f) {
                    invalidate_l1_cache();
                    if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f + 4 * hu_p) < q + 1 || !wr_ready() ||
                        !HCRED_OK(YH - 1, q)) {
                        return did;
                    }
                    piece_data<HALF, HP>(hu_p, part, cxs[CH], ny0 + YH - 1, part, htile);
                    hu_f = did = true;
                }
                if (!wr_ready()) {
                    return did;
                }
                piece_flag(hu_p, cxs[CH], ny0 + YH - 1, f, f);
                hu_f = false;
            } while (++hu_p < HP);
            hu_p = 0;
#ifndef WL_FNB
            noc_async_writes_flushed();
#endif
        }
        ++hu_i;
#ifndef WL_FNB
        hu_pub = hu_i;
#endif
        return true;
    };
    auto h_fwd_past = [&](uint32_t q) { return hu_pub >= 2 * q + 2 && (*fwd_word >> 16) >= 2 * q + 2; };
#elif defined(WL_HA2A)
    HFwd hn;
    hfwd_init<HALF>(hn, false, J / HALF, J % HALF);
    auto qhu = [&]() -> bool { return hfwd_step<NS, KTD>(hn, total_rows, cxs, ny0, hring, htile); };
    // (row q's slot is the source of both RISCs' forwards: to compute only once both have passed it)
    auto h_fwd_past = [&](uint32_t q) {
        return hn.q > q && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HA2A) > q;
    };
#else
    auto qhu = []() -> bool { return false; };
    auto h_fwd_past = [](uint32_t) { return true; };
#endif
#ifdef WL_KS
    // ---- QB (K split): push x row s (CB 0, KSL tiles) once its sender's flag (word s % 2 of the slot's 2) reads
    // >= s + 1; rows popped by compute -> XCRED word 0 here and word 1 at the partner
    uint32_t b_s = 0, x_popped_pages = 0, x_credited = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_x && cb_pages_reservable_at_back(xr_cb, KSL)) {
            invalidate_l1_cache();
            if (*KW(xflags + ((b_s % NX) * 2 + b_s % 2) * 16) >= b_s + 1) {
                cb_reserve_back(xr_cb, KSL);
                cb_push_back(xr_cb, KSL);
                ++b_s;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(xacked);
        x_popped_pages += (uint16_t)(a - x_last);
        x_last = a;
        if (x_popped_pages / KSL > x_credited) {
            x_credited = x_popped_pages / KSL;
            *KW(WL_KS_XCRED) = x_credited;
            noc_inline_dw_write(get_noc_addr(cxs[1 - J / HALF], ny0 + J % HALF, WL_KS_XCRED + 16), x_credited);
            did = true;
        }
        return did;
    };
    // ---- QLD (K split): push landing row q (CB 4, 20 tiles) once all HALF contributors' flags of slot q % D read
    // >= q + 1; rows popped by compute -> LCRED word Y here (the BRISC credits the rows below) and at the rows above
    uint32_t l_q = 0, l_pages = 0, l_cons = 0;
    const uintptr_t lacked = (uintptr_t)get_cb_tiles_acked_ptr(tt::CBIndex::c_4);
    uint16_t l_last = (uint16_t)reg_read(lacked);
    auto qld = [&]() -> bool {
        bool did = false;
        if (l_q < total_rows && cb_pages_reservable_at_back(tt::CBIndex::c_4, KS_LT)) {
            invalidate_l1_cache();
            bool all = true;
            for (uint32_t k = 0; k < HALF && all; ++k) {
                all = *KW(WL_KS_LFLAGS + ((l_q % WL_KS_D) * HALF + k) * 16) >= l_q + 1;
            }
            if (all) {
                cb_reserve_back(tt::CBIndex::c_4, KS_LT);
                cb_push_back(tt::CBIndex::c_4, KS_LT);
                ++l_q;
                did = true;
            }
        }
        const uint16_t a = (uint16_t)reg_read(lacked);
        l_pages += (uint16_t)(a - l_last);
        l_last = a;
        if (l_pages / KS_LT > l_cons) {
            l_cons = l_pages / KS_LT;
            *KW(WL_KS_LCRED + (J % HALF) * 16) = l_cons;
            for (uint32_t o = 0; o < J % HALF; ++o) {
                noc_inline_dw_write(get_noc_addr(cxs[J / HALF], ny0 + o, WL_KS_LCRED + (J % HALF) * 16), l_cons);
            }
            did = true;
        }
        return did;
    };
    // ---- QPU (K split): partials of row q to the owners above (o < Y, NOC1) once the BRISC published PAVAIL >= q + 1
    // (row q at slot q % PD of CB 18), once o consumed row q - D; then PDONE = q + 1
#ifdef WL_KS_PALL
    uint32_t pu_q = total_rows, pu_o = 0;  // (the BRISC sends every contribution)
#else
    uint32_t pu_q = 0, pu_o = 0;
#endif
    auto qpu = [&]() -> bool {
        if (pu_q >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*KW(WL_KS_SCR + 4 * 16) < pu_q + 1) {
            return false;
        }
        const uint32_t Cc = J / HALF, Yc = J % HALF;
        if (pu_o == 0) {
            *KW(WL_KS_SCR + 3 * 16) = pu_q + 1;
        }
        const uint32_t src = WL_KS_PRING + (pu_q % WL_KS_PD) * KS_NW * WL_KS_PTILE, sl = pu_q % WL_KS_D;
        bool did = false;
#ifdef WL_KS_BAL
        constexpr uint32_t PU_N = HALF - WL_KS_BAL;  // (the rows 1 .. PU_N up, wrapping round on NOC1)
#else
        const uint32_t PU_N = Yc;  // (the rows above)
#endif
        while (pu_o < PU_N) {
#ifdef WL_KS_BAL
            const uint32_t o = (Yc + HALF - 1 - pu_o) % HALF;
#else
            const uint32_t o = pu_o;
#endif
            if (pu_q >= WL_KS_D && *KW(WL_KS_LCRED + o * 16) < pu_q - WL_KS_D + 1) {
                return did;
            }
            const uint32_t oy = ny0 + o;
            noc_async_write(
                src + 2 * o * WL_KS_PTILE,
                get_noc_addr(cxs[Cc], oy, WL_KS_LAND + (sl * KS_LT + 2 * Yc) * WL_KS_PTILE),
                2 * WL_KS_PTILE);
            noc_async_write(WL_KS_SCR + 3 * 16, get_noc_addr(cxs[Cc], oy, WL_KS_LFLAGS + (sl * HALF + Yc) * 16), 4);
            ++pu_o;
            did = true;
        }
        noc_async_writes_flushed();
        *KW(WL_KS_SCR + 5 * 16) = pu_q + 1;  // PDONE
        ++pu_q;
        pu_o = 0;
        return true;
    };
    auto ks_done = [&]() { return l_q >= total_rows && l_cons >= total_rows && pu_q >= total_rows; };
#else
    // ---- QB: x ring pushes + credits
    uint32_t b_s = 0, x_popped_pages = 0, x_credited = 0, w_e = 0, w_base = 0;
    const uintptr_t xacked = (uintptr_t)get_cb_tiles_acked_ptr(xr_cb);
    uint16_t x_last = (uint16_t)reg_read(xacked);
    auto qb = [&]() -> bool {
        bool did = false;
        if (b_s < total_x && cb_pages_reservable_at_back(xr_cb, BK)) {
            invalidate_l1_cache();
#ifdef WL_NOEX
            if (true) {
#else
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + (b_s % NX) * 16 + XL) >= b_s + 1) {
#endif
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
        invalidate_l1_cache();
#ifdef WL_XHOP
        // hop credits: this core's freed count (blocks consumed and forwarded) to the cores that write into it - its
        // own word 3 (its BRISC's origin sends), the row above's word 0, the row below's word 1, the partner's word 2
        {
            const uint32_t x0 = x_credited;
            while (x_credited < x_popped_pages / BK && fwd_past(x_credited)) {
                ++x_credited;
            }
            if (x_credited != x0) {
                const uint32_t cx = J / HALF, cy = J % HALF;
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 3 * 16) = x_credited;
                if (cy > 0) {
                    noc_inline_dw_write(get_noc_addr(cxs[cx], ny0 + cy - 1, xprog + 0 * 16), x_credited);
                }
                if (cy + 1 < HALF) {
                    noc_inline_dw_write(get_noc_addr(cxs[cx], ny0 + cy + 1, xprog + 1 * 16), x_credited);
                }
                noc_inline_dw_write(get_noc_addr(cxs[1 - cx], ny0 + cy, xprog + 2 * 16), x_credited);
                did = true;
            }
        }
#endif
        while (x_credited < x_popped_pages / BK && fwd_past(x_credited)) {
            const uint32_t nxt = x_credited + NX;
            if (nxt < total_x) {
                while (nxt >= w_base + rows_of(w_e) * BPR) {
                    w_base += rows_of(w_e) * BPR;
                    ++w_e;
                }
#ifdef WL_BCAST
                const uint32_t owner = 0;  // (core 0 multicasts every block)
#else
                const uint32_t ol = nxt - w_base, owner = (ol % 2) * HALF + (ol / 2) % HALF;
#endif
#ifndef WL_NOEX
                noc_inline_dw_write(
                    get_noc_addr(cxs[owner / HALF], ny0 + owner % HALF, xprog + J * 16), x_credited + 1);
#endif
            }
            ++x_credited;
            did = true;
        }
        return did;
    };
    auto qld = []() -> bool { return false; };
    auto qpu = []() -> bool { return false; };
    auto ks_done = []() { return true; };
#endif
    // ---- QH: h ring rows (WL_HCRED: and this core's consumed count to its column and its partner)
    uint32_t h_q = 0;
#ifdef WL_HCRED
    uint32_t hc_pages = 0, hc_sent = 0;
    const uintptr_t hacked = (uintptr_t)get_cb_tiles_acked_ptr(hr_cb);
    uint16_t hc_last = (uint16_t)reg_read(hacked);
#endif
    auto qh = [&]() -> bool {
#ifdef WL_HCRED
        {
            const uint16_t a = (uint16_t)reg_read(hacked);
            hc_pages += (uint16_t)(a - hc_last);
            hc_last = a;
            if (hc_pages / KTD > hc_sent) {
                hc_sent = hc_pages / KTD;
                const uint32_t cc = J / HALF, yy = J % HALF;
                for (uint32_t o = 0; o < HALF; ++o) {
                    if (o == yy) {
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HCRED + yy * 16) = hc_sent;
                    } else {
                        noc_inline_dw_write(get_noc_addr(cxs[cc], ny0 + o, WL_HCRED + yy * 16), hc_sent);
                    }
                }
                noc_inline_dw_write(get_noc_addr(cxs[1 - cc], ny0 + yy, WL_HCRED + HALF * 16), hc_sent);
                return true;
            }
        }
#endif
        if (h_q >= total_rows || !cb_pages_reservable_at_back(hr_cb, KTD)) {
            return false;
        }
        invalidate_l1_cache();
#if defined(WL_HA2A)
        for (uint32_t t = 0; t < KTD; ++t) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HA2A_F + ((h_q % NS) * KTD + t) * 16) < h_q + 1) {
                return false;
            }
        }
#elif !defined(WL_NOEX)
        for (uint32_t c = 0; c < 2; ++c) {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + ((h_q % NS) * 2 + c) * 16 + HL) < h_q + 1) {
                return false;
            }
        }
#endif
        if (!h_fwd_past(h_q)) {  // (WL_HCHAIN: the row's slot is read for hops until both forwarders passed it)
            return false;
        }
        cb_reserve_back(hr_cb, KTD);
        cb_push_back(hr_cb, KTD);
        ++h_q;
        return true;
    };
    uint32_t mx[4] = {0, 0, 0, 0}, tot[4] = {0, 0, 0, 0};
    // ---- QL: column 1's h leader multicasts (rows q with q % 10 == Y; see the BRISC's Q3)
    [[maybe_unused]] const uint32_t C = J / HALF, Y = J % HALF;
    [[maybe_unused]] const uint32_t harr = get_semaphore(SEM_HARR);
    const uint32_t hscr_addr = get_semaphore(SEM_SCRH);
    volatile tt_l1_ptr uint32_t* hscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hscr_addr);
#ifdef WL_BCAST
    // core GC - 1 (NOC1's multicast start corner): row q once its GC tiles are in ring slot q % NS (the slot's count,
    // GC a use) and row q - 1's last flag has landed here (one h multicast in flight). The row goes to the other cores
    // only (it is already in place here, and a loopback copy could land after a producer's next tile for the slot),
    // unlinked; then both column flags to every core including this one, from the same command buffer, so each lands
    // after the row.
    const bool mch = J == GC - 1;
    uint32_t lq = 0;
    auto ql = [&]() -> bool {
        if (!mch || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_BCAST + (lq % NS) * 16) < GC * (lq / NS + 1)) {
            return false;
        }
        if (lq > 0 &&
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + (((lq - 1) % NS) * 2 + 1) * 16) < lq) {
            return false;
        }
        const uint32_t row = hring + (lq % NS) * KTD * htile;
        *hscr = lq + 1;
        noc_async_write_multicast(row, get_noc_multicast_addr(mx0, my0, mx1, my1, row), KTD * htile, GC - 1, false);
        for (uint32_t c = 0; c < 2; ++c) {
            noc_async_write_multicast_loopback_src(
                hscr_addr, get_noc_multicast_addr(mx0, my0, mx1, my1, hflags + ((lq % NS) * 2 + c) * 16), 4, GC,
                false);
        }
        noc_async_writes_flushed();
        ++lq;
        return true;
    };
#else
    const bool mch = (MC_COL_MASK >> C) & 1;
#if defined(WL_NOEX) || defined(WL_HA2A)
    uint32_t led = 0, lq = total_rows;
#else
    uint32_t led = 0, lq = Y;
#endif
    auto ql = [&]() -> bool {
        if (!((MC_COL_MASK >> C) & 1) || lq >= total_rows) {
            return false;
        }
        invalidate_l1_cache();
#ifdef WL_HARRW
        for (uint32_t y = 0; y < HALF; ++y) {  // (see the BRISC's Q2)
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HARRW + y * 16) < led + 1) {
                return false;
            }
        }
#else
        if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(harr) < HALF * (led + 1)) {
            return false;
        }
#endif
        const uint32_t part = hring + ((lq % NS) * KTD + C * HALF) * htile;
        for (uint32_t p = 0; p < 4; ++p) {  // (all four words: see piece_hop)
            hscr[p] = lq + 1;
        }
#if defined(WL_HCHAIN)
        // (chain start: the part is in place here; flag it locally, send it to the partner column, and the forwarders
        // take it on from both)
        if (!HCRED_OK(HALF, lq)) {  // (the partner has consumed row lq - NS)
            return false;
        }
        const uint32_t hf = hflags + ((lq % NS) * 2 + C) * 16;
        for (uint32_t p = 0; p < HP; ++p) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hf)[p] = lq + 1;
        }
        for (uint32_t p = 0; p < HP; ++p) {
            piece_hop<HALF, HP>(p, part, cxs[1 - C], ny0 + Y, part, htile, hscr_addr, hf);
#ifdef WL_JUMP
            for (uint32_t c2 = 0; c2 < 2; ++c2) {
                piece_hop<HALF, HP>(p, part, cxs[c2], ny0 + (Y + HALF / 2) % HALF, part, htile, hscr_addr, hf);
            }
#endif
        }
#elif defined(WL_UCAST)
        group_write_with_flag<HALF>(
            cxs, ny0, J, part, part, HALF * htile, hscr_addr, hflags + ((lq % NS) * 2 + C) * 16);
#else
        mcast_with_flag(
            part,
            get_noc_multicast_addr(mx0, my0, mx1, my1, part),
            HALF * htile,
            GC,
            hscr_addr,
            get_noc_multicast_addr(mx0, my0, mx1, my1, hflags + ((lq % NS) * 2 + C) * 16));
#endif
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };
#endif
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
#ifdef WL_YBARRIER
            noc_async_write_barrier();
#else
            noc_async_writes_flushed();  // (the CB pages only must be free again: no wait for the DRAM acks)
#endif
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
#ifdef WL_HCRED
#define HC_DONE (hc_sent >= total_rows)
#else
#define HC_DONE true
#endif
#if defined(WL_KS) && !defined(WL_BCAST)
#define KS_A_DONE (ka_q >= total_rows || !mcx)
#else
#define KS_A_DONE (a_n >= nexp_g || !mcx)
#endif
    uint32_t nbz[4] = {0, 0, 0, 0};
    const uint32_t t_start = wall();
    uint32_t iter = 0, t_fwd = 0;  // (t_fwd: loop time in the up-forwarders, part of tot[2])
    while (true) {
#ifdef WL_FPRIO
#define WL_FU() (fu |= qu() | qhu())  // (see the BRISC)
#else
#define WL_FU()
#endif
        bool fu = qu() | qhu();
        const uint32_t t0 = wall();
        const bool y = qy();
        WL_FU();
        const bool l = ql();
        WL_FU();
        bool a = false;
        if (++iter % WL_WPOLL == 0) {  // (the weight queues every WL_WPOLL turns: the x / h queues pace compute)
            invalidate_l1_cache();
            a = qw() | qf() | qr();
            WL_FU();
        }
        const uint32_t t1 = wall();
        const bool b = qa();
        WL_FU();
        const uint32_t t2 = wall();
        bool c = qb() | qld() | qpu();
#ifdef WL_WTWICE
        c |= qw() | qf() | qr();  // (the weight queues again, after the x / landing / partial queues)
#endif
        const uint32_t tf = wall();
        const bool fu2 = qu() | qhu();
        fu |= fu2;
        const uint32_t t3 = wall();
        if (fu2) {  // (busy time only)
            t_fwd += t3 - tf;
        }
        const bool d = qh();
        const uint32_t t4 = wall();
        const uint32_t dd[4] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3};
        nbz[0] += (y || l || a) ? t1 - t0 : 0;  // (busy time: QY / QL / weights, QA, QB, QH; the forwarders in t_fwd)
        nbz[1] += b ? t2 - t1 : 0;
        nbz[2] += c ? tf - t2 : 0;
        nbz[3] += d ? t4 - t3 : 0;
        for (uint32_t i = 0; i < 4; ++i) {
            tot[i] += dd[i];
            mx[i] = dd[i] > mx[i] ? dd[i] : mx[i];
        }
        if (!fu && fu_s >= total_x && fu_pub >= total_x && hu_i >= total_hd && hu_pub >= total_hd && !y && !l && !a &&
            !b && !c && !d && ks_done() && HC_DONE && drained >= total_rows && (!mch || lq >= total_rows) &&
            w_n >= nexp_g && w_dn >= nexp_g && freed >= nexp_g && slots_done() && KS_A_DONE && b_s >= total_x &&
            x_credited >= total_x && h_q >= total_rows) {
            break;
        }
#ifdef WL_IDLE
        if (!fu && !y && !l && !a && !b && !c && !d) {  // (an idle turn: back off on a register, not L1)
            const uint32_t w0 = wall();
            while (wall() - w0 < WL_IDLE) {
            }
        }
#endif
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
    st[478] = t_fwd;
    for (uint32_t i = 0; i < 4; ++i) {
        st[480 + i] = nbz[i];
    }
    st[484] = wall() - t_start;
    st[485] = t_start;
    st[59] = n_credit;
    for (uint32_t i = 0; i < 4; ++i) {
        st[470 + i] = tot[i];
        st[474 + i] = mx[i];
    }
}
