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
//     9 next chip direction 0, 10 next chip direction 1, 11 this core's xy, 12 x addr, 13 idx addr, 14 counts addr,
//     15 table addr, 16 own slices addr, 17 destination lists addr (WL_Y_RM), then the mux client RT args (17 words) of
//     direction 0 and of direction 1
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 GC, 5 G, 6 NS, 7 NX, 8 hflags, 9 xring, 10 xflags, 11 xprog, 12-15 group rect
//     for NOC1, 16 x tile bytes, 17 counts L1, 18 table L1, 19 hring, 20 h tile bytes, 21 warr (arrivals a slot,
//     16 B apart), 22 ORMAX (output tile rows a group), 23 HT, 24 wgu ring,
//     25 wd ring, 26 cred (2K words), 27 FREEW (K words), 28 CHT, 29 W tile bytes, 30-34 mux client CT args
//     (buffers, buffer bytes, status, termination signal, clients), then TensorAccessorArgs(out), (x), (idx),
//     (counts), (table), (own slices), (destination lists), then RMAX, x row bytes, NEXP, table page bytes, idx
//     scratch L1 (Q1's token ids at +0, QY's destinations at +256), D (own chunk reads in flight)
// WL_NO_FABRIC (one chip, every expert its own): no mux connections and no mux client RT args, nothing forwarded
// and no credits sent; CT 30-34 are placeholders.
// WL_Y_RM: row-major outputs. The compute pack-untilizes each down row tile (32 rows x DCOLS * 32 columns, row after
// row in CB 17); each of its rows goes to its own row of the row-major output, at this core's columns: the row's entry
// in the expert's destination list (same layout as the token lists: expert e's page, entry p for its p-th token).
// Rows past the expert's count are not written. With WL_Y_POSTED the writes are posted (no acks: DRAM reads and
// acknowledged DRAM writes on one NoC can hang the chip), fenced at the end (WL_Y_FENCE).
#include "api/dataflow/dataflow_api.h"
#ifndef WL_NO_FABRIC
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;
#endif

constexpr uint32_t xr_cb = tt::CBIndex::c_0;
constexpr uint32_t w_cb = tt::CBIndex::c_1;
constexpr uint32_t hr_cb = tt::CBIndex::c_2;
constexpr uint32_t wd_cb = tt::CBIndex::c_3;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
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
#elif defined(WL_NOC_SPLIT)
// WL_NOC_SPLIT: no multicasts here (the BRISC sends them all on NOC0); this RISC does every DRAM read of the core on
// NOC1 instead: the counts and the slot table (counts word 0 then marks them loaded), the token-row gather (Q1) and
// the own experts' weights (QO, transaction ids 1..D), see wl_writer.cpp
constexpr uint32_t MC_COL_MASK = 0;
#else
constexpr uint32_t MC_COL_MASK = 2;
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
#ifndef WL_NOC_SPLIT
FORCE_INLINE void mcast_with_flag(
    uint32_t src, uint64_t dst, uint32_t bytes, uint32_t n, uint32_t flag_src, uint64_t flag_dst) {
#ifdef WL_MCAST_LINKED
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, true);
#else
    noc_async_write_multicast_loopback_src(src, dst, bytes, n, false);
#endif
    noc_async_write_multicast_loopback_src(flag_src, flag_dst, 4, n, false);
}
#endif

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
    constexpr auto xargs = TensorAccessorArgs<oargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr auto targs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto wargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto dargs = TensorAccessorArgs<wargs.next_compile_time_args_offset()>();
    constexpr uint32_t RB = dargs.next_compile_time_args_offset();
    [[maybe_unused]] constexpr uint32_t RMAX = get_compile_time_arg_val(RB), xrow = get_compile_time_arg_val(RB + 1);
    [[maybe_unused]] constexpr uint32_t NEXP = get_compile_time_arg_val(RB + 2), TB = get_compile_time_arg_val(RB + 3);
    [[maybe_unused]] constexpr uint32_t idx_l1 = get_compile_time_arg_val(RB + 4);
    [[maybe_unused]] constexpr uint32_t D = get_compile_time_arg_val(RB + 5);
    constexpr uint32_t BPR = KT / BK, HALF = GC / 2;
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;  // packets of a slice (gate / up first)
    static_assert(WGU % CHT == 0 && WDN % CHT == 0);

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    [[maybe_unused]] const uint16_t my_mesh = get_arg_val<uint32_t>(8);
    [[maybe_unused]] const uint8_t nchip[2] = {(uint8_t)get_arg_val<uint32_t>(9), (uint8_t)get_arg_val<uint32_t>(10)};
    const uint32_t myxy = get_arg_val<uint32_t>(11);
    [[maybe_unused]] const uint32_t mex = myxy & 0xFF, mey = myxy >> 8;

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
#ifdef WL_NOC_SPLIT
    const auto xs = TensorAccessor(xargs, get_arg_val<uint32_t>(12), xrow);
    const auto is = TensorAccessor(iargs, get_arg_val<uint32_t>(13), RMAX * 32 * 4);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(16), wtile);
    {
        const auto cs = TensorAccessor(cargs, get_arg_val<uint32_t>(14), NEXP * 4);
        const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(15), TB);
        noc_async_read_page(0, cs, counts_l1 + 64);
        noc_async_read_page(g, ts, table_l1);
        noc_async_read_barrier();
        counts[0] = 1;
    }
#endif
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

#ifndef WL_NO_FABRIC
    // ---- mux connections (index d: to the next chip in direction d), headers
    tt::tt_fabric::WorkerToFabricMuxSender<mux_nbuf> mux[2];
    bool master[2];
    uint32_t term_sync[2], tmx[2], tmy[2];
    uint8_t muxx[2], muxy[2];
    for (uint32_t d = 0; d < 2; ++d) {
        size_t a_ = 18 + 17 * d;
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
#endif

    // ---- slots: current occupant (its table entry cached in registers), its packets sent a direction, computed
    volatile tt_l1_ptr uint32_t* warr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(warr_addr);
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* cred = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cred_addr);
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
    // packet u of slot s in this core's L1 (same on every chip)
    [[maybe_unused]] auto slot_addr = [&](uint32_t s, uint32_t u) {
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
    [[maybe_unused]] uint32_t rr = 0, n_fwd = 0, n_credit = 0;
#ifdef WL_NO_FABRIC
    auto qf = [&]() -> bool { return false; };  // (one chip: no slot occupant goes on to another chip)
#else
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
                    get_noc_addr(mex, mey, slot_addr(s2, u)), get_noc_addr(mex, mey, warr_addr + s2 * 16), 1, true});
#endif
            ++sent[s][d];
            ++n_fwd;
            rr = rr + t + 1;
            return true;
        }
        return false;
    };
#endif
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
#ifndef WL_NO_FABRIC
            if (nk >= 2 && !mux[1 - (nk - 2)].edm_has_space_for_packet()) {
                continue;
            }
#endif
            noc_async_writes_flushed();  // (every forward of the slot has left L1 before it is refilled)
            if (nk == 1) {
                freew[s * 4] = next;  // (the BRISC reads the chip's own next expert into it)
            }
#ifndef WL_NO_FABRIC
            else if (nk >= 2) {
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
#endif
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
#elif defined(WL_NOC_SPLIT)
    const bool mcx = false;  // (the BRISC multicasts every x block on NOC0)
    auto qa = []() -> bool { return false; };
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
#endif

    // ---- Q1 (WL_NOC_SPLIT): the token-row gather on NOC1, a state machine (0 idle, 1 ids requested, 2 rows
    // requested) so the fabric and x / h queues of this RISC never wait on a read barrier
    uint32_t g_n = nexp_g, g_state = 0;
    [[maybe_unused]] uint32_t g_l = L0;
#ifdef WL_NOC_SPLIT
    g_n = 0;
    auto q1 = [&]() -> bool {
        if (g_state == 1) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            const uint32_t k = (g_l % BPR) * BK;
            volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
            cb_reserve_back(xrm_cb, BK);
            uint32_t l1 = get_write_ptr(xrm_cb);
            for (uint32_t i = 0; i < 32; ++i, l1 += BK * 64) {
                noc_async_read(xs.get_noc_addr(idx[i]) + k * 64, l1, BK * 64);
            }
            g_state = 2;
            return true;
        }
        if (g_state == 2) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            cb_push_back(xrm_cb, BK);
            g_l += GC;
            g_state = 0;
            return true;
        }
        while (g_n < nexp_g && g_l >= rows_of(g_n) * BPR) {
            ++g_n;
            g_l = L0;
        }
        if (g_n >= nexp_g || !cb_pages_reservable_at_back(xrm_cb, BK)) {
            return false;
        }
#ifdef WG_NO_GATHER
        cb_reserve_back(xrm_cb, BK);
        cb_push_back(xrm_cb, BK);
        g_l += GC;
        return true;
#endif
        noc_async_read(is.get_noc_addr(order[g_n] & 0xFFFF) + (g_l / BPR) * 32 * 4, idx_l1, 32 * 4);
        g_state = 1;
        return true;
    };
#else
    auto q1 = []() -> bool { return false; };
#endif
    // ---- QO (WL_NOC_SPLIT): the own experts' weights from DRAM into their slots on NOC1 (chunk reads with
    // transaction ids 1..D, retired in order without a barrier); a slot's next own occupant is read once QR freed it
    [[maybe_unused]] uint32_t o_s = 0, o_k[K] = {}, o_issued = 0, o_landed = 0, n_own = 0;
    [[maybe_unused]] bool o_busy = false;
#ifdef WL_NOC_SPLIT
    auto entry = [&](uint32_t s, uint32_t k) { return tab + tab[1 + K + s] + 4 * k; };
    auto next_own = [&]() {  // a slot whose next own occupant may be read now (its previous one freed), slots in turn
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
#ifdef WL_NO_OWN_READ
        o_issued = o_landed = PK;
        warr[s * 4] = PK * k + PK;
        did = true;
#else
        if (o_landed < o_issued) {
            const uint32_t trid = 1 + o_landed % D;
            if (ncrisc_noc_read_with_transaction_id_flushed(noc_index, trid)) {
                ++o_landed;
                warr[s * 4] = PK * k + o_landed;
                did = true;
            }
        }
        if (o_issued < PK && o_issued - o_landed < D) {
            const uint32_t u = o_issued;
            const uint32_t q = (entry(s, k)[3] * GC + J) * PK + u;  // chunk q of the chip's own slices
            noc_async_read_set_trid(1 + u % D);
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), slot_addr(s, u), CHT * wtile);
            noc_async_read_set_trid(0);
            ++o_issued;
            did = true;
        }
#endif
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
#else
    auto qo = []() -> bool { return false; };
    auto qo_done = []() { return true; };
#endif
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
#ifdef WL_BCAST
                const uint32_t owner = 0;  // (core 0 multicasts every block)
#else
                const uint32_t ol = nxt - w_base, owner = (ol % 2) * HALF + (ol / 2) % HALF;
#endif
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
    [[maybe_unused]] const uint32_t C = J / HALF, Y = J % HALF;
    [[maybe_unused]] const uint32_t harr = get_semaphore(SEM_HARR);
    const uint32_t hscr_addr = get_semaphore(SEM_SCRH);
    volatile tt_l1_ptr uint32_t* hscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hscr_addr);
#if defined(WL_BCAST) && !defined(WL_NOC_SPLIT)
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
#elif defined(WL_NOC_SPLIT)
    const bool mch = false;  // (the BRISC multicasts every h row on NOC0)
    const uint32_t lq = total_rows;
    auto ql = []() -> bool { return false; };
#else
    const bool mch = (MC_COL_MASK >> C) & 1;
    uint32_t led = 0, lq = Y;
    auto ql = [&]() -> bool {
        if (!((MC_COL_MASK >> C) & 1) || lq >= total_rows) {
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
#endif
    // ---- Q2 (WL_BCAST with WL_NOC_SPLIT): this core's tile of h row hs into core 0's ring slot hs % NS on NOC1
    // (west / north, inside the group), with its own transaction id; once acknowledged, the slot's arrival count at
    // core 0 + 1 and the tile freed. The slot holds row hs - NS until that row's multicast has left core 0: its last
    // flag landing here shows it.
#if defined(WL_BCAST) && defined(WL_NOC_SPLIT)
    constexpr uint32_t h_cb = tt::CBIndex::c_16;
    constexpr uint32_t H_TRID = 0xE;  // (own chunk reads use 1..D)
    static_assert(D < H_TRID);
    uint32_t hs = 0;
    bool hs_wait = false;
    auto q2 = [&]() -> bool {
        if (hs_wait) {
            if (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, H_TRID)) {
                return false;
            }
            noc_semaphore_inc(get_noc_addr(cxs[0], ny0, WL_BCAST + (hs % NS) * 16), 1);
            cb_pop_front(h_cb, 1);
            ++hs;
            hs_wait = false;
            return true;
        }
        if (hs >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
        if (hs >= NS) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + ((hs % NS) * 2 + 1) * 16) <
                hs - NS + 1) {
                return false;
            }
        }
        noc_async_write_one_packet_with_trid(
            get_read_ptr(h_cb),
            get_noc_addr(cxs[0], ny0, hring + ((hs % NS) * KTD + C * HALF + Y) * htile),
            htile,
            H_TRID);
        noc_async_write_set_trid(0);
        hs_wait = true;
        return true;
    };
#else
    const uint32_t hs = total_rows;
    const bool hs_wait = false;
    auto q2 = []() -> bool { return false; };
#endif
#ifdef WG_Y_NCRISC
    // ---- QY: down outputs (CB 17) to DRAM, on NOC1 (the BRISC's NOC0 keeps the DRAM reads): the group's last
    // expert's rows at tile row g ORMAX, or (WL_OUT_ALL) every expert's, packed in order from g ORMAX, or (WL_Y_RM)
    // each row of a row tile to its destination row
    constexpr uint32_t y_cb = tt::CBIndex::c_17;
    uint32_t drained = 0, d_n = 0, d_r = 0;
#ifdef WL_Y_RM
    // a state machine (0: next row tile, 1: its 32 destinations requested, 2: its rows written): the destinations are
    // read on their own transaction id before the row tile is computed, and the pages are freed once the writes left
    constexpr uint32_t Y_TRID = 0xD;  // (own chunk reads use 1..D)
    static_assert(D < Y_TRID);
    constexpr uint32_t seg = DCOLS * 64;  // a row's bytes at this core's columns
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(7), HT * 64);
    const auto ds = TensorAccessor(dargs, get_arg_val<uint32_t>(17), RMAX * 32 * 4);
    constexpr uint32_t dst_l1 = idx_l1 + 256;
    volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_l1);
    uint32_t y_state = 0;
    auto qy = [&]() -> bool {
        if (y_state == 2) {
#ifdef WL_Y_POSTED
            if (!ncrisc_noc_posted_writes_sent(noc_index)) {
#else
            if (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
#endif
                return false;
            }
            cb_pop_front(y_cb, DCOLS);
            ++d_r;
            ++drained;
            y_state = 0;
            return true;
        }
        if (y_state == 1) {
            if (!ncrisc_noc_read_with_transaction_id_flushed(noc_index, Y_TRID) ||
                !cb_pages_available_at_front(y_cb, DCOLS)) {
                return false;
            }
            invalidate_l1_cache();
            const uint32_t m = counts[16 + (order[d_n] & 0xFFFF)] - d_r * 32;
            uint32_t l1 = get_read_ptr(y_cb);
#ifndef WL_Y_NOWRITE
            for (uint32_t r = 0; r < (m < 32 ? m : 32); ++r, l1 += seg) {
#ifdef WL_Y_POSTED
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(l1, os.get_noc_addr(dst[r], J * seg), seg);
#else
                noc_async_write(l1, os.get_noc_addr(dst[r], J * seg), seg);
#endif
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
        noc_async_read(ds.get_noc_addr(order[d_n] & 0xFFFF) + d_r * 128, dst_l1, 128);
        noc_async_read_set_trid(0);
        y_state = 1;
        return true;
    };
#else
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, get_arg_val<uint32_t>(7), obytes);
    [[maybe_unused]] uint32_t d_base = 0;
    bool y_sent = false;  // a row's writes issued, its CB 17 pages still held
    auto qy = [&]() -> bool {
        if (y_sent) {
            // the row's pages are freed once the NoC has read them out; no wait for the DRAM acks, so this RISC's
            // other queues keep running while the writes complete
            if (!ncrisc_noc_nonposted_writes_sent(noc_index)) {
                return false;
            }
            y_sent = false;
            cb_pop_front(y_cb, DCOLS);
            ++d_r;
            ++drained;
            return true;
        }
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
            y_sent = true;
            return true;
        }
        cb_pop_front(y_cb, DCOLS);
        ++d_r;
        ++drained;
        return true;
    };
#endif
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
        const bool r = q1() | qo() | q2();
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
        if (!y && !l && !a && !b && !c && !d && !r && g_n >= nexp_g && g_state == 0 && qo_done() &&
            hs >= total_rows && !hs_wait &&
            drained >= total_rows && (!mch || lq >= total_rows) && w_n >= nexp_g &&
            w_dn >= nexp_g && freed >= nexp_g && slots_done() && (a_n >= nexp_g || !mcx) &&
            b_s >= total_x && x_credited >= total_x && h_q >= total_rows) {
            break;
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
#ifdef WL_Y_POSTED
    noc_async_posted_writes_flushed();
#ifdef WL_Y_FENCE
    // the posted row writes' fence: an acknowledged 4-B write into every DRAM bank (the output's spare rows from
    // WL_Y_FENCE, nobody reads them) after them on the same NoC and VC; its acks mean every row landed
    for (uint32_t i = 0; i < NUM_DRAM_BANKS; ++i) {
        noc_async_write(dst_l1, os.get_noc_addr(WL_Y_FENCE + i), 4);
    }
    noc_async_write_barrier();
#endif
#endif
#ifndef WL_NO_FABRIC
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
#endif
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(6));
    st[48] = t_cred;
    st[49] = t_empty;
    st[58] = n_fwd;
    st[59] = n_credit;
#ifdef WL_NOC_SPLIT
    st[56] = n_own;
#endif
    for (uint32_t i = 0; i < 4; ++i) {
        st[470 + i] = tot[i];
        st[474 + i] = mx[i];
    }
}
