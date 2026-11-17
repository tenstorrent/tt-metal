// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ws_expert lane design, group BRISC (NOC0): the group writer of ws_group (wg_writer.cpp) in this chip's compute order
// (ws_lane_sched), plus the chip's own experts' weights. Loads this group's slot tables (a DRAM page per group, see
// wl_reader.cpp) and the expert counts into L1 (counts word 0 = ready), then a non-blocking loop over:
//   QO own weights: the slots' occupants of kind own, in each slot's order: once the NCRISC freed the slot's previous
//      occupant (FREEW + 16 s >= k), read the core's slice of the expert from DRAM (PK chunks of CHT tiles, one bank
//      each, ws_fab.slice_layout over the chip's own experts) into the slot, setting the slot's arrival word to
//      PK k + chunks landed as they complete (no fabric arrival can be in flight for the slot meanwhile: the slot's
//      next occupant's sender waits for its credit, sent only after this one is computed and forwarded).
//      WL_NO_OWN_READ (a diagnostic): no reads, each own occupant's slot is marked arrived unread.
//   Q1 x gather, Q2 h out, Q3 column 0 leader multicasts, QX column 0 x multicasts: as in wg_writer.cpp, the experts
//      taken in the order table's order (the down outputs go to DRAM from the NCRISC).
// RT: 0 j, 1 g, 2 order length, 3 col x NOC (c 0), 4 col x NOC (c 1), 5 row 0 y NOC, 6 out addr, 7 x addr, 8 idx addr,
//     9 counts addr, 10 stamps, 11 table addr, 12 own slices addr
// CT: 0 KT, 1 BK, 2 KTD, 3 DCOLS, 4 GC, 5 G, 6 NS, 7 hring, 8 hflags, 9-12 group rect NOC0, 13 h tile bytes, 14 HT,
//     15 RMAX, 16 x row bytes, 17 counts L1, 18 idx scratch L1, 19 NEXP, 20 NX, 21 xring, 22 xflags, 23 xprog,
//     24 x tile bytes, 25 table L1, 26 table page bytes, 27 wgu ring, 28 wd ring, 29 warr, 30 FREEW, 31 CHT,
//     32 W tile bytes, 33 D (chunk reads in flight), then TensorAccessorArgs(out), (x), (idx), (counts), (table),
//     (own slices)
#include "api/dataflow/dataflow_api.h"

// WL_HCRED: h ring slot credits. Every core's NCRISC writes the count of h rows its compute has consumed (and so its
// forwarders passed) into word Y of each core of its column and word HALF of its partner (16 B words at WL_HCRED); a
// write of row q into core X's ring slot (an h tile to its leader, a chain start to the partner, a hop) waits for X's
// count >= q - NS + 1. Without it the ring depth NS alone had to cover the cores' skew (PCC dips in the K split).
#ifdef WL_HCRED
#define HCRED_OK(word, q) \
    ((q) < NS || *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HCRED + (word) * 16) >= (q) - NS + 1)
#else
#define HCRED_OK(word, q) true
#endif
#ifdef WL_HA2A
#include "wl_ha2a.hpp"
#endif
constexpr uint32_t xrm_cb = tt::CBIndex::c_7;
constexpr uint32_t h_cb = tt::CBIndex::c_16;
constexpr uint32_t y_cb = tt::CBIndex::c_17;
constexpr uint32_t xst_cb = tt::CBIndex::c_8;
constexpr uint32_t SEM_HARR = 1, SEM_SCR = 2, SEM_SCRX = 4;
#ifndef WL_K
#define WL_K 3
#endif
constexpr uint32_t K = WL_K;  // weight slots a core
// columns whose x blocks (X_COLS) and h leader rows (H_COLS) this BRISC multicasts on NOC0 (bit c)
#if defined(WL_BCAST)
// WL_BCAST: one multicaster a group a NoC, each at its NoC's multicast start corner, so no multicast leaves the group
// rectangle on its way to that corner. x: every producer sends its tilized blocks on NOC1 (west / north) into core 0's
// funnel slots (WL_XIN, WL_XF slots, filled flags at WL_XINF), and core 0's BRISC multicasts every block on NOC0 in
// stream order, one at a time. h: every core sends its tile of a row on NOC0 (east / south) to core GC - 1, whose
// NCRISC multicasts the whole row on NOC1, one at a time. WL_BCAST: NS per-slot h arrival counters (16 B apart).
constexpr uint32_t X_COLS = 0, H_COLS = 0;
#elif defined(WL_MC_NOC1)
constexpr uint32_t X_COLS = 0, H_COLS = 0;  // (column 0's multicasts go out from its NCRISC on NOC1)
#else
constexpr uint32_t X_COLS = 1, H_COLS = 1;
#endif

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

// WL_XCHAIN: x blocks travel the group as a unicast chain instead of a fan-out (each link carries a block once, as a
// multicast would). Block s of origin core (c, r) (r = (l / 2) % HALF for the expert's block l): the origin copies it
// into its own ring slot and sends it to its partner (1 - c, r) (column 0's BRISC on NOC0 east, column 1's NCRISC on
// NOC1 west); then in both columns every core of a row >= r passes it one row down from its BRISC on NOC0, and every
// core of a row <= r one row up from its NCRISC on NOC1, from the ring slot, flag word included (value s + 1, stable
// until the slot's credit). A core credits block s to its producer only once both its forwarders are past s (the
// BRISC's count in semaphore 0), so no slot is overwritten while it is read for a hop.
FORCE_INLINE void x_hop(
    uint32_t src, uint32_t x, uint32_t y, uint32_t dst, uint32_t bytes, uint32_t flag_src, uint32_t flag_dst) {
    noc_async_write(src, get_noc_addr(x, y, dst), bytes);
    noc_async_write(flag_src, get_noc_addr(x, y, flag_dst), 4);  // (same buffer, NoC and VC: after the payload)
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
    constexpr uint32_t hring = get_compile_time_arg_val(7);
    constexpr uint32_t hflags = get_compile_time_arg_val(8);
    constexpr uint32_t gx0 = get_compile_time_arg_val(9), gy0 = get_compile_time_arg_val(10);
    constexpr uint32_t gx1 = get_compile_time_arg_val(11), gy1 = get_compile_time_arg_val(12);
    constexpr uint32_t htile = get_compile_time_arg_val(13);
    constexpr uint32_t HT = get_compile_time_arg_val(14);
    constexpr uint32_t RMAX = get_compile_time_arg_val(15);
    constexpr uint32_t xrow = get_compile_time_arg_val(16);
    constexpr uint32_t counts_l1 = get_compile_time_arg_val(17);
    constexpr uint32_t idx_l1 = get_compile_time_arg_val(18);
    constexpr uint32_t NEXP = get_compile_time_arg_val(19);
    constexpr uint32_t NX = get_compile_time_arg_val(20);
    constexpr uint32_t xring = get_compile_time_arg_val(21), xflags = get_compile_time_arg_val(22);
    constexpr uint32_t xprog = get_compile_time_arg_val(23), xtile = get_compile_time_arg_val(24);
    constexpr uint32_t table_l1 = get_compile_time_arg_val(25), TB = get_compile_time_arg_val(26);
    constexpr uint32_t wgu_ring = get_compile_time_arg_val(27), wd_ring = get_compile_time_arg_val(28);
    constexpr uint32_t warr_addr = get_compile_time_arg_val(29), freew_addr = get_compile_time_arg_val(30);
    constexpr uint32_t CHT = get_compile_time_arg_val(31), wtile = get_compile_time_arg_val(32);
    constexpr uint32_t D = get_compile_time_arg_val(33);
    constexpr auto oargs = TensorAccessorArgs<34>();
    constexpr auto xargs = TensorAccessorArgs<oargs.next_compile_time_args_offset()>();
    constexpr auto iargs = TensorAccessorArgs<xargs.next_compile_time_args_offset()>();
    constexpr auto cargs = TensorAccessorArgs<iargs.next_compile_time_args_offset()>();
    constexpr auto targs = TensorAccessorArgs<cargs.next_compile_time_args_offset()>();
    constexpr auto wargs = TensorAccessorArgs<targs.next_compile_time_args_offset()>();
    constexpr uint32_t BPR = KT / BK;
    constexpr uint32_t WGU = KT * 2, WDN = KTD * DCOLS;
    constexpr uint32_t GU_PK = WGU / CHT, PK = (WGU + WDN) / CHT;
    static_assert(D >= 1 && D <= 8);
    constexpr uint32_t HALF = GC / 2;  // cores a column

    const uint32_t J = get_arg_val<uint32_t>(0), g = get_arg_val<uint32_t>(1), nexp_g = get_arg_val<uint32_t>(2);
    const uint32_t cxs[2] = {get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4)};
    const uint32_t ny0 = get_arg_val<uint32_t>(5);
    const uint32_t oaddr = get_arg_val<uint32_t>(6), xaddr = get_arg_val<uint32_t>(7);
    const uint32_t iaddr = get_arg_val<uint32_t>(8), caddr = get_arg_val<uint32_t>(9);
    const uint32_t C = J / HALF, Y = J % HALF;
    const uint32_t obytes = get_tile_size(y_cb);
    const auto os = TensorAccessor(oargs, oaddr, obytes);
    const auto xs = TensorAccessor(xargs, xaddr, xrow);
    const auto is = TensorAccessor(iargs, iaddr, RMAX * 32 * 4);
    const auto cs = TensorAccessor(cargs, caddr, NEXP * 4);
    const auto ts = TensorAccessor(targs, get_arg_val<uint32_t>(11), TB);
    const auto ws = TensorAccessor(wargs, get_arg_val<uint32_t>(12), wtile);

    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counts_l1);
    volatile tt_l1_ptr uint32_t* tab = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(table_l1);
    volatile tt_l1_ptr uint32_t* order = tab + 16;
    noc_async_read_page(0, cs, counts_l1 + 64);
    noc_async_read_page(g, ts, table_l1);
    noc_async_read_barrier();
    counts[0] = 1;
    auto rows_of = [&](uint32_t n) { return (counts[16 + (order[n] & 0xFFFF)] + 31) / 32; };
    uint32_t total_rows = 0;
    for (uint32_t n = 0; n < nexp_g; ++n) {
        total_rows += rows_of(n);
    }

#ifdef WL_KS
    // ---- Q1 (K split): this core's x rows q of the stream with q % 2 == C (its partner gathers the others), whole:
    // 32 token rows x KSL K-tiles from K tile KSL Y (768 B reads) into CB 7. Pipelined: the token ids of KS_IB rows of
    // an expert come in one read (idx scratch, KS_IB x 128 B), and the data reads of as many own rows as CB 7 has room
    // for are issued before one wait (state 0 idle, 1 ids requested, 2 rows requested)
    constexpr uint32_t KSL = WL_KS, KS_IB = 8;
    auto KW = [](uint32_t a) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(a); };
    [[maybe_unused]] const uint32_t L0 = 0;
    const uint32_t koff = KSL * Y * 64;
    uint32_t g_n = 0, g_r = 0, g_q = 0, g_state = 0, g_pend = 0, ib_n = 0xFFFFFFFF, ib_r0 = 0, ib_cnt = 0;
    // (CB 7 is a ring of WL_KS_XRM rows: a batch's rows after the first wrap round it)
    const uint32_t xrm_base = get_write_ptr(xrm_cb);
    constexpr uint32_t XRM_ROW = KSL * 2048, XRM_SIZE = WL_KS_XRM * XRM_ROW;
    auto own_row = [&]() {  // advance (g_n, g_r, g_q) to this core's next row (or past the last expert)
        while (true) {
            while (g_n < nexp_g && g_r >= rows_of(g_n)) {
                ++g_n;
                g_r = 0;
            }
            if (g_n >= nexp_g || g_q % 2 == C) {
                return;
            }
            ++g_r;
            ++g_q;
        }
    };
    auto q1 = [&]() -> bool {
        if (g_state != 0) {
            if (!ncrisc_noc_reads_flushed(noc_index)) {
                return false;
            }
            if (g_state == 2) {
                cb_push_back(xrm_cb, KSL * g_pend);
                g_pend = 0;
            }
            g_state = 0;
            return true;
        }
        own_row();
        if (g_n >= nexp_g) {
            return false;
        }
        if (ib_n != g_n || g_r < ib_r0 ||
            g_r >= ib_r0 + ib_cnt) {  // ids of this row not in the buffer: read KS_IB rows'
            ib_n = g_n;
            ib_r0 = g_r;
            ib_cnt = rows_of(g_n) - g_r < KS_IB ? rows_of(g_n) - g_r : KS_IB;
            noc_async_read(is.get_noc_addr(order[g_n] & 0xFFFF) + g_r * 32 * 4, idx_l1, ib_cnt * 32 * 4);
            g_state = 1;
            return true;
        }
        // data reads of own rows while their ids are buffered and CB 7 has room
        while (g_n < nexp_g && ib_n == g_n && g_r >= ib_r0 && g_r < ib_r0 + ib_cnt &&
               cb_pages_reservable_at_back(xrm_cb, KSL * (g_pend + 1))) {
            cb_reserve_back(xrm_cb, KSL * (g_pend + 1));
            volatile tt_l1_ptr uint32_t* idx = KW(idx_l1 + (g_r - ib_r0) * 32 * 4);
            uint32_t l1 = xrm_base + (get_write_ptr(xrm_cb) - xrm_base + g_pend * XRM_ROW) % XRM_SIZE;
            for (uint32_t i = 0; i < 32; ++i, l1 += KSL * 64) {
                noc_async_read(xs.get_noc_addr(idx[i]) + koff, l1, KSL * 64);
            }
            ++g_pend;
            ++g_r;
            ++g_q;
            own_row();
        }
        if (g_pend) {
            g_state = 2;
            return true;
        }
        return false;
    };
#else
    // ---- Q1: a state machine (0 idle, 1 ids requested, 2 rows requested): the barriers would stall the x
    // multicasts this RISC also sends (column 1) - they went out 11-27K cycles late
    const uint32_t L0 = 2 * (J % HALF) + J / HALF;
    uint32_t g_n = 0, g_l = L0, g_state = 0;
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
#endif
    // ---- QO: own experts' weights from DRAM into their slots (chunk reads with transaction ids 1..D, in order)
    volatile tt_l1_ptr uint32_t* warr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(warr_addr);
    volatile tt_l1_ptr uint32_t* freew = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(freew_addr);
    auto entry = [&](uint32_t s, uint32_t k) { return tab + tab[1 + K + s] + 4 * k; };
    uint32_t o_s = 0, o_k[K] = {}, o_issued = 0, o_landed = 0, n_own = 0;
    bool o_busy = false;
    auto next_own = [&]() {  // a slot whose next own occupant may be read now (its previous one freed), slots in turn
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
#ifdef WL_NO_OWN_READ
        // (no DRAM read: the slot's chunks count as landed with whatever the slot holds)
        o_issued = o_landed = PK;
        warr[s * 4] = PK * k + PK;
        did = true;
#else
        if (o_landed < o_issued) {  // (retired only once done: a barrier would stall this RISC's x / h queues)
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
            const uint32_t dst = u < GU_PK ? wgu_ring + (s * WGU + u * CHT) * wtile
                                           : wd_ring + (s * WDN + (u - GU_PK) * CHT) * wtile;
            noc_async_read_set_trid(1 + u % D);
            noc_async_read(ws.get_noc_addr((q / 8) * CHT * 8 + q % 8), dst, CHT * wtile);
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
    // ---- Q2: h tile -> column leader (WL_BCAST: -> core 0, counted in the row's ring slot)
    uint32_t sent = 0;
    const uint32_t harr = get_semaphore(SEM_HARR);
    auto q2 = [&]() -> bool {
        if (sent >= total_rows || !cb_pages_available_at_front(h_cb, 1)) {
            return false;
        }
#ifdef WL_HA2A
        {  // (own tile -> own ring slot + its flag; the forwarders take it on: see wl_ha2a.hpp)
            invalidate_l1_cache();
            if (sent >= NS && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HCRED + Y * 16) < sent - NS + 1) {
                return false;
            }
            const uint32_t s = sent % NS, t = C * HALF + Y;
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HA2A + 16) = sent + 1;
            noc_async_write(get_read_ptr(h_cb), get_noc_addr(cxs[C], ny0 + Y, hring + (s * KTD + t) * htile), htile);
            noc_async_write(WL_HA2A + 16, get_noc_addr(cxs[C], ny0 + Y, WL_HA2A_F + (s * KTD + t) * 16), 4);
            noc_async_writes_flushed();
            cb_pop_front(h_cb, 1);
            ++sent;
            return true;
        }
#endif
#ifdef WL_NOEX
        cb_pop_front(h_cb, 1);
        ++sent;
        return true;
#endif
        invalidate_l1_cache();
        if (!HCRED_OK(sent % HALF, sent)) {  // (the row's leader has consumed row sent - NS)
            return false;
        }
#ifdef WL_BCAST
        // core GC - 1's ring slot sent % NS holds row sent - NS until that row's multicast has been read out of it:
        // its last flag landing here shows the whole row has left that core
        if (sent >= NS) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hflags + ((sent % NS) * 2 + 1) * 16) < sent - NS + 1) {
                return false;
            }
        }
        const uint32_t lx = cxs[1], ly = ny0 + HALF - 1, arr = WL_BCAST + (sent % NS) * 16;
#else
        const uint32_t lx = cxs[C], ly = ny0 + sent % HALF, arr = harr;
#endif
        noc_async_write(
            get_read_ptr(h_cb), get_noc_addr(lx, ly, hring + ((sent % NS) * KTD + C * HALF + Y) * htile), htile);
#if defined(WL_HARRW) && !defined(WL_BCAST)
        // (this core's (sent / HALF + 1)-th tile to that leader, counted in its word Y there: the word after the tile
        // through the same command buffer, so it lands after it; flushed, no barrier: the source only must be free.
        // The arrival semaphore, unused, holds the value.)
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(harr) = sent / HALF + 1;
        noc_async_write(harr, get_noc_addr(lx, ly, WL_HARRW + Y * 16), 4);
        noc_async_writes_flushed();
        (void)arr;
#else
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(lx, ly, arr), 1);
#endif
        cb_pop_front(h_cb, 1);
        ++sent;
        return true;
    };
    // ---- Q3: leader multicasts (rows q with q % 10 == Y)
#if defined(WL_NOEX) || defined(WL_HA2A)
    uint32_t led = 0, lq = total_rows;
#else
    uint32_t led = 0, lq = Y;
#endif
    const uint32_t scr_addr = get_semaphore(SEM_SCR);
    volatile tt_l1_ptr uint32_t* scr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr_addr);
    constexpr uint32_t ngroup = (gx1 - gx0 + 1) * (gy1 - gy0 + 1);
    auto q3 = [&]() -> bool {
        if (!((H_COLS >> C) & 1) || lq >= total_rows) {  // (column 1's leaders multicast on NOC1 from the NCRISC)
            return false;
        }
        invalidate_l1_cache();
#ifdef WL_HARRW
        for (uint32_t y = 0; y < HALF; ++y) {
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
            scr[p] = lq + 1;
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
            piece_hop<HALF, HP>(p, part, cxs[1 - C], ny0 + Y, part, htile, scr_addr, hf);
#ifdef WL_JUMP
            for (uint32_t c2 = 0; c2 < 2; ++c2) {
                piece_hop<HALF, HP>(p, part, cxs[c2], ny0 + (Y + HALF / 2) % HALF, part, htile, scr_addr, hf);
            }
#endif
        }
#elif defined(WL_UCAST)
        group_write_with_flag<HALF>(cxs, ny0, J, part, part, HALF * htile, scr_addr, hflags + ((lq % NS) * 2 + C) * 16);
#else
        mcast_with_flag(
            part,
            get_noc_multicast_addr(gx0, gy0, gx1, gy1, part),
            HALF * htile,
            ngroup,
            scr_addr,
            get_noc_multicast_addr(gx0, gy0, gx1, gy1, hflags + ((lq % NS) * 2 + C) * 16));
#endif
        noc_async_writes_flushed();
        ++led;
        lq += HALF;
        return true;
    };
    // ---- Q4: down outputs
#ifdef WG_Y_NCRISC
    // (the down outputs go to DRAM from the NCRISC on NOC1: this RISC's NOC0 only reads DRAM)
    uint32_t drained = total_rows, d_n = 0, d_r = 0;
    auto q4 = [&]() -> bool { return false; };
#else
    uint32_t drained = 0, d_n = 0, d_r = 0;
    auto q4 = [&]() -> bool {
        if (drained >= total_rows || !cb_pages_available_at_front(y_cb, DCOLS)) {
            return false;
        }
        while (d_r >= rows_of(d_n)) {
            ++d_n;
            d_r = 0;
        }
        if (d_n + 1 == nexp_g) {
            uint32_t l1 = get_read_ptr(y_cb);
            const uint32_t row = g * RMAX + d_r;
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
#endif
    auto wall = []() { return *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L); };
    // ---- QX: x multicasts of column 0 (NOC0)
    [[maybe_unused]] uint32_t a_n = 0, a_l = L0, a_base = 0;
    const uint32_t xscr_addr = get_semaphore(SEM_SCRX);
    volatile tt_l1_ptr uint32_t* xscr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xscr_addr);
#ifdef WL_BCAST
    // (core 0) block s of the stream: from this core's staging if it is one of its own (stream index within the
    // expert l with l % GC == 0), else from funnel slot s % WL_XF once its producer has filled it; then the ring credit
    // and one-at-a-time checks of the other modes, and the multicast to every core including this one
    const uint32_t total_x = total_rows * BPR;
    uint32_t xs_s = 0, xs_n = 0, xs_base = 0;
    auto qx = [&]() -> bool {
        if (J != 0 || xs_s >= total_x) {
            return false;
        }
        const uint32_t s = xs_s;
        while (xs_n < nexp_g && s >= xs_base + rows_of(xs_n) * BPR) {
            xs_base += rows_of(xs_n) * BPR;
            ++xs_n;
        }
        const bool own = (s - xs_base) % GC == 0;
        invalidate_l1_cache();
        uint32_t src;
        if (own) {
            if (!cb_pages_available_at_front(xst_cb, BK)) {
                return false;
            }
            src = get_read_ptr(xst_cb);
        } else {
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_XINF + (s % WL_XF) * 16) < s + 1) {
                return false;
            }
            src = WL_XIN + (s % WL_XF) * BK * xtile;
        }
        if (s >= NX) {
            for (uint32_t c = 0; c < GC; ++c) {
                if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + c * 16) < s - NX + 1) {
                    return false;
                }
            }
        }
        if (s >= 1 && *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + ((s - 1) % NX) * 16) < s) {
            return false;
        }
        const uint32_t xs = s % NX;
        *xscr = s + 1;
#ifdef WL_UCAST
        group_write_with_flag<HALF>(cxs, ny0, J, src, xring + xs * BK * xtile, BK * xtile, xscr_addr, xflags + xs * 16);
#else
        mcast_with_flag(
            src,
            get_noc_multicast_addr(gx0, gy0, gx1, gy1, xring + xs * BK * xtile),
            BK * xtile,
            ngroup,
            xscr_addr,
            get_noc_multicast_addr(gx0, gy0, gx1, gy1, xflags + xs * 16));
#endif
        noc_async_writes_flushed();
        if (own) {
            cb_pop_front(xst_cb, BK);
        }
        ++xs_s;
        return true;
    };
    auto x_done = [&]() { return J != 0 || xs_s >= total_x; };
#elif defined(WL_KS)
    // ---- QX (K split, column 0; column 1's on the NCRISC, NOC1 west): this core's tilized x row s (CB 8, KSL tiles)
    // -> ring slot s % NX of itself and its partner (payload + flag word C of the slot's 2, from one command buffer),
    // once both consumed row s - NX (XCRED words 0 / 1, kept by the NCRISCs)
    uint32_t kx_q = C;
    auto x_done = [&]() { return C != 0 || kx_q >= total_rows; };
    auto qx = [&]() -> bool {
#ifdef WL_KS_XDIRECT
        // (compute tilized row kx_q straight into its ring slot here and signalled it with one CB 8 page: send it to
        // the partner, then flag it here - after the send has left, so compute cannot reuse the slot under it)
        if (C != 0 || kx_q >= total_rows || !cb_pages_available_at_front(xst_cb, 1)) {
            return false;
        }
        if (kx_q >= NX) {
            invalidate_l1_cache();
            if (*KW(WL_KS_XCRED + 16) < kx_q - NX + 1) {
                return false;
            }
        }
        {
            const uint32_t a = xring + (kx_q % NX) * KSL * xtile, fl = xflags + ((kx_q % NX) * 2 + C) * 16;
            *KW(WL_KS_SCR) = kx_q + 1;
            noc_async_write(a, get_noc_addr(cxs[1 - C], ny0 + Y, a), KSL * xtile);
            noc_async_write(WL_KS_SCR, get_noc_addr(cxs[1 - C], ny0 + Y, fl), 4);
            noc_async_writes_flushed();
            *KW(fl) = kx_q + 1;
            cb_pop_front(xst_cb, 1);
            kx_q += 2;
            return true;
        }
#endif
        if (C != 0 || kx_q >= total_rows || !cb_pages_available_at_front(xst_cb, KSL)) {
            return false;
        }
        if (kx_q >= NX) {
            invalidate_l1_cache();
            if (*KW(WL_KS_XCRED) < kx_q - NX + 1 || *KW(WL_KS_XCRED + 16) < kx_q - NX + 1) {
                return false;
            }
        }
        const uint32_t dst = xring + (kx_q % NX) * KSL * xtile, fl = xflags + ((kx_q % NX) * 2 + C) * 16;
        *KW(WL_KS_SCR) = kx_q + 1;
        const uint32_t src = get_read_ptr(xst_cb);
        noc_async_write(src, get_noc_addr(cxs[C], ny0 + Y, dst), KSL * xtile);
        noc_async_write(WL_KS_SCR, get_noc_addr(cxs[C], ny0 + Y, fl), 4);
        noc_async_write(src, get_noc_addr(cxs[1 - C], ny0 + Y, dst), KSL * xtile);
        noc_async_write(WL_KS_SCR, get_noc_addr(cxs[1 - C], ny0 + Y, fl), 4);
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, KSL);
        kx_q += 2;
        return true;
    };
    // ---- QP (K split): row q's 20 partial tiles (CB 18; local pair p = partial tiles 2 p, 2 p + 1) to their owners
    // (C, o) for o >= Y (rows below on NOC0, itself): landing slot q % WL_KS_D at contributor Y + flag q + 1, once o
    // consumed row q - D (LCRED word o). PAVAIL (scratch word 4) = q + 1 tells the NCRISC (rows above); the row is
    // popped once its PDONE (word 5) reaches q + 1. QC: this core's landing credits (LCRED word Y, kept by the
    // NCRISC) to the rows below.
    constexpr uint32_t p_cb = tt::CBIndex::c_18, KS_NW = 2 * HALF, KS_LT = 2 * HALF;
#if defined(WL_KS_PALL)
    // (WL_KS_PALL: this BRISC sends every contribution, the rows above too - NOC0 wraps round the column - so the
    // NCRISC's loop stays with the weights; no PDONE handshake)
    constexpr uint32_t KP_N = HALF;
#elif defined(WL_KS_BAL)
    // (WL_KS_BAL: by distance, not direction - this BRISC to itself and the next WL_KS_BAL - 1 rows down, wrapping
    // round the column's NoC ring on NOC0; the NCRISC to the rest, up on NOC1: every core the same share)
    constexpr uint32_t KP_N = WL_KS_BAL;
#else
    const uint32_t KP_N = HALF - Y;  // (itself and the rows below)
#endif
    uint32_t kp_q = 0, kp_d = 0, kc_sent = 0;
    bool kp_av = false;
    auto qp = [&]() -> bool {
        if (kp_q >= total_rows) {
            return false;
        }
        if (!kp_av) {
            if (!cb_pages_available_at_front(p_cb, KS_NW)) {
                return false;
            }
            kp_av = true;
            *KW(WL_KS_SCR + 16) = kp_q + 1;      // this RISC's partial flag source
            *KW(WL_KS_SCR + 4 * 16) = kp_q + 1;  // PAVAIL
        }
        bool did = false;
        const uint32_t src = get_read_ptr(p_cb), sl = kp_q % WL_KS_D;
        while (kp_d < KP_N) {
            const uint32_t kp_o = (Y + kp_d) % HALF;
            if (kp_q >= WL_KS_D) {
                invalidate_l1_cache();
                if (*KW(WL_KS_LCRED + kp_o * 16) < kp_q - WL_KS_D + 1) {
                    return did;
                }
            }
            const uint32_t oy = ny0 + kp_o;
            noc_async_write(
                src + 2 * kp_o * WL_KS_PTILE,
                get_noc_addr(cxs[C], oy, WL_KS_LAND + (sl * KS_LT + 2 * Y) * WL_KS_PTILE),
                2 * WL_KS_PTILE);
            noc_async_write(WL_KS_SCR + 16, get_noc_addr(cxs[C], oy, WL_KS_LFLAGS + (sl * HALF + Y) * 16), 4);
            ++kp_d;
            did = true;
        }
        invalidate_l1_cache();
#ifndef WL_KS_PALL
        if (*KW(WL_KS_SCR + 5 * 16) < kp_q + 1) {  // PDONE
            return did;
        }
#endif
        noc_async_writes_flushed();
        cb_pop_front(p_cb, KS_NW);
        ++kp_q;
        kp_d = 0;
        kp_av = false;
        return true;
    };
    auto qc = [&]() -> bool {
        invalidate_l1_cache();
        const uint32_t c = *KW(WL_KS_LCRED + Y * 16);
        if (c <= kc_sent) {
            return false;
        }
        for (uint32_t o = Y + 1; o < HALF; ++o) {
            noc_inline_dw_write(get_noc_addr(cxs[C], ny0 + o, WL_KS_LCRED + Y * 16), c);
        }
        kc_sent = c;
        return true;
    };
    auto ks_done = [&]() { return kp_q >= total_rows && kc_sent >= total_rows; };
#else
    auto x_done = [&]() { return !((X_COLS >> C) & 1) || a_n >= nexp_g; };
    auto qx = [&]() -> bool {
        if (!((X_COLS >> C) & 1)) {
            return false;
        }
        while (a_n < nexp_g && a_l >= rows_of(a_n) * BPR) {
            a_base += rows_of(a_n) * BPR;
            ++a_n;
            a_l = L0;
        }
        if (a_n >= nexp_g || !cb_pages_available_at_front(xst_cb, BK)) {
            return false;
        }
        const uint32_t s = a_base + a_l;
#ifdef WL_NOEX
        cb_pop_front(xst_cb, BK);
        a_l += GC;
        return true;
#endif
#ifdef WG_TRACE0
        volatile tt_l1_ptr uint32_t* tst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(10));
        if (s >= WG_TRACE0 && s < WG_TRACE0 + 40 && tst[320 + s - WG_TRACE0] == 0) {
            tst[320 + s - WG_TRACE0] = wall();
        }
#endif
#ifdef WL_XHOP
        // (hop credits: only this core's own slot and its partner's, the two places this block is written from here)
        if (s >= NX) {
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 3 * 16) < s - NX + 1 ||
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 2 * 16) < s - NX + 1) {
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
#ifdef WG_TRACE0
                    if (s >= WG_TRACE0 && s < WG_TRACE0 + 40) {
                        tst[384 + s - WG_TRACE0] = c + 1;
                        tst[448 + s - WG_TRACE0] = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + c * 16);
                    }
#endif
                    return false;
                }
            }
        }
#if WG_XORDER
        if (s >= WG_XORDER) {  // in stream order (see the NCRISC)
            invalidate_l1_cache();
            if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xflags + ((s - WG_XORDER) % NX) * 16 + XL) <
                s - WG_XORDER + 1) {
                return false;
            }
        }
#endif
        const uint32_t xs = s % NX;
        for (uint32_t p = 0; p < 4; ++p) {  // (all four words: see piece_hop)
            xscr[p] = s + 1;
        }
#if defined(WL_XCHAIN)
        {
            const uint32_t src = get_read_ptr(xst_cb), dst = xring + xs * BK * xtile, xf = xflags + xs * 16;
            for (uint32_t p = 0; p < XP; ++p) {  // (piece by piece to every entry, so they all start forwarding early)
                piece_hop<BK, XP>(p, src, cxs[C], ny0 + Y, dst, xtile, xscr_addr, xf);
                piece_hop<BK, XP>(p, src, cxs[1 - C], ny0 + Y, dst, xtile, xscr_addr, xf);
#ifdef WL_JUMP
                for (uint32_t c2 = 0; c2 < 2; ++c2) {
                    piece_hop<BK, XP>(p, src, cxs[c2], ny0 + (Y + HALF / 2) % HALF, dst, xtile, xscr_addr, xf);
                }
#endif
            }
        }
#elif defined(WL_UCAST)
        group_write_with_flag<HALF>(
            cxs, ny0, J, get_read_ptr(xst_cb), xring + xs * BK * xtile, BK * xtile, xscr_addr, xflags + xs * 16);
#else
    mcast_with_flag(
        get_read_ptr(xst_cb),
        get_noc_multicast_addr(gx0, gy0, gx1, gy1, xring + xs * BK * xtile),
        BK * xtile,
        ngroup,
        xscr_addr,
        get_noc_multicast_addr(gx0, gy0, gx1, gy1, xflags + xs * 16));
#endif
        noc_async_writes_flushed();
        cb_pop_front(xst_cb, BK);
#ifdef WG_TRACE0
        if (s >= WG_TRACE0 && s < WG_TRACE0 + 40) {
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(10))[128 + s - WG_TRACE0] = wall();
        }
#endif
        a_l += GC;
        return true;
    };
#endif
#if !defined(WL_KS) || defined(WL_BCAST)
    auto qp = []() -> bool { return false; };
    auto qc = []() -> bool { return false; };
    auto ks_done = []() { return true; };
#endif
    // chain forwarders' progress for the NCRISC (slot credits / h rows to compute): semaphore 0, x blocks walked in
    // the low 16 bits, h row parts in the high 16. Each forwarder call passes on every piece that has landed, up to the
    // end of its block / part. WL_FNB: no blocking flush after a hop; what the forwarders walked is published once this
    // RISC's writes have all left, checked without blocking (else: flushed after each block / part, then published).
    volatile tt_l1_ptr uint32_t* fwd_word = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    ASSERT(total_rows * BPR < 65536);
#ifdef WL_XCHAIN
    const uint32_t total_xd = total_rows * BPR;
    uint32_t fd_s = Y + 1 < HALF ? 0 : total_xd, fd_n = 0, fd_base = 0, fd_p = 0;
    bool fd_f = false;  // (piece fd_p's payload issued, its flag not yet)
#else
    const uint32_t total_xd = 0, fd_s = 0;
#endif
#ifdef WL_HCHAIN
    const uint32_t total_hd = 2 * total_rows;
    uint32_t hd_i = Y + 1 < HALF ? 0 : total_hd, hd_p = 0;
    bool hd_f = false;
#else
    const uint32_t total_hd = 0, hd_i = 0;
#endif
    uint32_t fd_pub = fd_s, hd_pub = hd_i;
    auto pub = [&]() { *fwd_word = (hd_pub << 16) | fd_pub; };
    pub();
    [[maybe_unused]] auto fnb = [&]() -> bool {
#ifdef WL_FNB
        if ((fd_pub != fd_s || hd_pub != hd_i) && ncrisc_noc_nonposted_writes_sent(noc_index)) {
            fd_pub = fd_s;
            hd_pub = hd_i;
            pub();
            return true;
        }
#endif
        return false;
    };
#ifdef WL_XCHAIN
    // ---- QD: x chain, one row down (NOC0): blocks in stream order, those whose origin row is <= this row
    auto qd = [&]() -> bool {
        bool did = fnb();
        if (fd_s >= total_xd) {
            return did;
        }
        while (fd_s >= fd_base + rows_of(fd_n) * BPR) {
            fd_base += rows_of(fd_n) * BPR;
            ++fd_n;
        }
        if (chain_down<HALF>(Y, ((fd_s - fd_base) / 2) % HALF)) {
            const uint32_t xs = fd_s % NX, xf = xflags + xs * 16;
            do {
                if (!fd_f) {
                    invalidate_l1_cache();
                    if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xf + 4 * fd_p) < fd_s + 1) {
                        return did;
                    }
#ifdef WL_XHOP
                    if (fd_p == 0 && fd_s >= NX &&
                        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(xprog + 0 * 16) <
                            fd_s - NX + 1) {  // (the row below has freed the slot)
                        return did;
                    }
#endif
                    if (!wr_ready()) {
                        return did;
                    }
                    piece_data<BK, XP>(
                        fd_p, xring + xs * BK * xtile, cxs[C], ny0 + Y + 1, xring + xs * BK * xtile, xtile);
                    fd_f = did = true;
                }
                if (!wr_ready()) {
                    return did;
                }
                piece_flag(fd_p, cxs[C], ny0 + Y + 1, xf, xf);
                fd_f = false;
            } while (++fd_p < XP);
            fd_p = 0;
#ifndef WL_FNB
            noc_async_writes_flushed();
#endif
        }
        ++fd_s;
#ifndef WL_FNB
        fd_pub = fd_s;
        pub();
#endif
        return true;
    };
#else
    auto qd = []() -> bool { return false; };
#endif
#ifdef WL_HCHAIN
    // ---- QHD: h chain, one row down (NOC0): row parts i = 2 q + c in order, those whose leader row q % HALF is <=
    // this row; the walked count in semaphore 0's high half (the NCRISC gives row q to compute only once both
    // forwarders passed it)
    auto qhd = [&]() -> bool {
        bool did = fnb();
        if (hd_i >= total_hd) {
            return did;
        }
        const uint32_t q = hd_i / 2, c = hd_i % 2;
        if (chain_down<HALF>(Y, q % HALF)) {
            const uint32_t f = hflags + ((q % NS) * 2 + c) * 16;
            const uint32_t part = hring + ((q % NS) * KTD + c * HALF) * htile;
            do {
                if (!hd_f) {
                    invalidate_l1_cache();
                    if (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(f + 4 * hd_p) < q + 1 || !wr_ready() ||
                        !HCRED_OK(Y + 1, q)) {
                        return did;
                    }
                    piece_data<HALF, HP>(hd_p, part, cxs[C], ny0 + Y + 1, part, htile);
                    hd_f = did = true;
                }
                if (!wr_ready()) {
                    return did;
                }
                piece_flag(hd_p, cxs[C], ny0 + Y + 1, f, f);
                hd_f = false;
            } while (++hd_p < HP);
            hd_p = 0;
#ifndef WL_FNB
            noc_async_writes_flushed();
#endif
        }
        ++hd_i;
#ifndef WL_FNB
        hd_pub = hd_i;
        pub();
#endif
        return true;
    };
#else
    auto qhd = []() -> bool { return false; };
#endif
#ifdef WL_HA2A
    HFwd hb;
    hfwd_init<HALF>(hb, true, C, Y);
    auto qhb = [&]() -> bool {
        const bool d = hfwd_step<NS, KTD>(hb, total_rows, cxs, ny0, hring, htile);
        if (d) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(WL_HA2A) = hb.q;  // (rows forwarded: for the NCRISC's QH)
        }
        return d;
    };
#define HB_DONE (hb.q >= total_rows)
#else
    auto qhb = []() -> bool { return false; };
#define HB_DONE true
#endif
    uint32_t tq[5] = {0, 0, 0, 0, 0}, tmx[5] = {0, 0, 0, 0, 0}, t_fwd = 0, t_own = 0;  // (forwarders, own weights)
    uint32_t tbz[5] = {0, 0, 0, 0, 0};  // (busy time of qx / q1 / q2 / q3 / q4: calls that did something)
    uint32_t t_ks = 0;                  // (busy time of the K split's partial sends and landing credits)
    const uint32_t t_start = wall();
    while (true) {
        // WL_FPRIO: the chain forwarders between every other queue (a hop waiting a whole turn of the loop costs every
        // core further down the chain)
#ifdef WL_FPRIO
#define WL_FW() (fdw |= qd() | qhd())
#else
#define WL_FW()
#endif
        const uint32_t tf0 = wall();
        bool fdw = qd() | qhd() | qhb();
        const uint32_t tf1 = wall();
        const bool o = qo();
        if (fdw) {  // (busy time only: a call that did something)
            t_fwd += tf1 - tf0;
        }
        if (o) {
            t_own += wall() - tf1;
        }
        WL_FW();
        uint32_t t0 = wall();
        const bool x = qx();
        WL_FW();
        uint32_t t1 = wall();
        const bool a = q1();
        WL_FW();
        uint32_t t2 = wall();
        const bool b = q2();
        WL_FW();
        uint32_t t3 = wall();
        const bool c = q3();
        WL_FW();
        uint32_t t4 = wall();
        const bool d = q4();
        uint32_t t5 = wall();
        const bool kp = qp() | qc();
        if (kp) {
            t_ks += wall() - t5;
        }
        tq[0] += t1 - t0;
        tq[1] += t2 - t1;
        tq[2] += t3 - t2;
        tq[3] += t4 - t3;
        tq[4] += t5 - t4;
        const uint32_t dd[5] = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4};
        const bool bz[5] = {x, a, b, c, d};
        for (uint32_t i = 0; i < 5; ++i) {
            tmx[i] = dd[i] > tmx[i] ? dd[i] : tmx[i];
            tbz[i] += bz[i] ? dd[i] : 0;
        }
        const bool mch = (H_COLS >> C) & 1;
        if (!fdw && fd_s >= total_xd && fd_pub >= total_xd && hd_i >= total_hd && hd_pub >= total_hd && !o && !x &&
            !a && !b && !c && !d && !kp && ks_done() && HB_DONE && g_n >= nexp_g && sent >= total_rows &&
            (!mch || lq >= total_rows) && drained >= total_rows && x_done() && g_state == 0 && qo_done()) {
            break;
        }
#ifdef WL_IDLE
        if (!fdw && !o && !x && !a && !b && !c && !d && !kp) {  // (an idle turn: back off on a register, not L1)
            const uint32_t w0 = wall();
            while (wall() - w0 < WL_IDLE) {
            }
        }
#endif
    }
    noc_async_atomic_barrier();
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(10));
    for (uint32_t i = 0; i < 5; ++i) {
        st[50 + i] = tq[i];
        st[104 + i] = tmx[i];
    }
    st[56] = n_own;
    st[60] = t_fwd;
    st[61] = t_own;
    for (uint32_t i = 0; i < 5; ++i) {
        st[490 + i] = tbz[i];  // (not 62..: the compute's end stamp is 63)
    }
    st[495] = wall() - t_start;
    st[496] = t_start;
    st[497] = t_ks;
}
