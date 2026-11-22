// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention at R = 32 for U users of M rows (M5 K2a, bench/resident/m5/K2A_DESIGN_v2.md + addenda 1-2):
// 8 head cores (one per q head) and 60 unit cores per chip.
// d_k2a (M6b integration, bench/resident/m6/M6B_INTEGRATION_PLAN.md): a copy of resident_m5_3's k2a_common.hpp (frozen)
// with a drafter mode, kDraftW > 0 (ct 13: the DFlash2 drafter's sliding_window W): every layer sliding; a user block
// at p .. p + M - 1 sees positions [p - W, p + M - 1] in every row (DFlash2 _mask, non-causal); K / V from the
// slot-major drafter ring (no page tables); V passes through the head cores (no v_norm); roots per core <= 2 and
// partials per root <= ceil(W / 256) + 1 (d_k2a_sched.py GI0). kDraftW = 0 is K2a's behaviour.
//   head core h: its q head's 32 rows from the streamers (rx) -> q = rope(rmsnorm(q) w_q); sliding kv head g = h / 2:
//     h even also k = rope(rmsnorm(k) w_k), h odd v = rmsnorm(v); full: h = 0 k = rope(rmsnorm(kv) w_k), h = 1
//     v = rmsnorm(kv); K / V as BF8. The prepped Q goes to a DRAM staging buffer in the unit cores' packed layout, K / V
//     as BF8 tiles of all 32 rows; then READY on every unit core. O: the roots' rows of q head h land in cb_og; once
//     every live user's are in, one multicast of the Dt tiles into the streamers' o_in (NoC 1) and their sem_o.
//   unit core: units (user u, kv head g, absolute chunk: sliding 8 position tiles, full 64) and roots (the core holding
//     (u, g)'s tail unit: it splices u's new K / V rows into the tail tiles, merges (u, g)'s partials by chunk, sends O
//     rows to the head cores). Every unit core computes the same schedule from the feed rows at launch (NCRISC).
// Q2 (row invariance): every op is per row, units and sub-chunks sit on the absolute grid, masks come from absolute
// positions, the merge order is by absolute chunk.
// NoC ownership (rule N1): unit BRISC owns NoC 0 (K reads, tail K splice, partials), unit NCRISC NoC 1 (V, Q, tables,
// tail V splice, O rows); head BRISC NoC 0 (staging writes, READY), head NCRISC NoC 1 (reads, the O multicast).
#pragma once

#include <stdint.h>

namespace k2a {

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t M = get_compile_time_arg_val(1);  // rows per user
constexpr uint32_t num_streamers = get_compile_time_arg_val(2);
constexpr uint32_t eps_bits = get_compile_time_arg_val(3);
constexpr uint32_t inv_ds_bits = get_compile_time_arg_val(4);
constexpr uint32_t inv_df_bits = get_compile_time_arg_val(5);
constexpr uint32_t kGroupKinds = get_compile_time_arg_val(6);  // groups holding sliding (bits 0..5) / full (8..13)
constexpr uint32_t KR = get_compile_time_arg_val(7);           // cb_k pages (the K ring)
constexpr uint32_t VR = get_compile_time_arg_val(8);           // cb_v pages (the V ring)
constexpr uint32_t NP = get_compile_time_arg_val(9);           // cb_parts pages
constexpr uint32_t kNumUnits = get_compile_time_arg_val(10);   // unit cores (60)
constexpr uint32_t kPerHead = get_compile_time_arg_val(11);    // sliding unit cores per kv head (15)
constexpr uint32_t CQ = get_compile_time_arg_val(12);          // cb_q pages (Q slots x Dt, the larger kind's)
constexpr uint32_t kDraftW = get_compile_time_arg_val(13);     // drafter window (0: K2a)
constexpr bool kDraft = kDraftW > 0;
constexpr uint32_t kRingL0 = get_compile_time_arg_val(14);     // (drafter) the program's first layer in the ring

constexpr uint32_t R = 32;
constexpr uint32_t U = R / M;  // users per launch
static_assert(M == 1 || M == 2 || M == 4 || M == 8, "rows per user");
static_assert(!kDraft || M >= 2, "drafter blocks: 2, 4 or 8 rows (U <= 16: two roots per core per kv head)");
constexpr uint32_t kHeads = 8;  // q heads per chip (= head cores)
constexpr uint32_t kDts = 8, kDtf = 16;
constexpr uint32_t kBf8 = 1088, kBf16 = 2048;
constexpr uint32_t kPage = 128, kWindow = 1024;
constexpr uint32_t kChunkS = 8, kChunkF = 64;  // absolute chunks (position tiles)
constexpr uint32_t NVs = 9, NVf = 8;           // sub-chunk tiles: sliding = the whole unit (<= 8 + 1 straddling tail
                                               // tile), full = the absolute grid of 8
constexpr uint32_t kMaxChunksF = 5;            // full units per user: positions < 5 * 2048 (K2a; longer is K2b's)
// partials per sliding root (cb_sc / cb_diag pages: the root merge's per-partial tiles); drafter: the window's chunks
constexpr uint32_t kMaxKS = kDraft ? (kDraftW + 32 * kChunkS - 1) / (32 * kChunkS) + 1 : 5;
constexpr uint32_t kSC = kMaxKS > 8 ? kMaxKS : 8;
// drafter ring (d_ingest.py DrafterRing): per (slot, layer, kv head) kRingTiles position tiles of Dt BF8 tiles, all
// kRingLayers drafter layers whatever the program's (a program runs ring layers [kRingL0, kRingL0 + layers))
constexpr uint32_t kRingTiles = 72, kRingLayers = 5;
static_assert(!kDraft || kRingL0 + layers <= kRingLayers, "the program's layers within the ring's");
// a user's window + block spans <= W / 32 + 3 position tiles: no two of them share a ring tile
static_assert(!kDraft || kDraftW / 32 + 3 <= kRingTiles, "the window + block fit the ring");
constexpr uint16_t kNegBig = 0xce6e;           // bf16 -1e9 (as the engine's attention)

// packed Q layout of kv head g (QH q heads): user u's QH * M rows over T_u tiles of H_t heads, k users per tile
template <bool full>
struct Pack {
    static constexpr uint32_t QH = full ? 8 : 2;
    static constexpr uint32_t Ru = QH * M;
    static constexpr uint32_t Tu = (Ru + 31) / 32;   // Q tiles per user (2: full at M = 8)
    static constexpr uint32_t Ht = QH / Tu;          // heads per tile
    static constexpr uint32_t k = 32 / (Ht * M);     // users per tile
    static constexpr uint32_t tiles = (U / k) * Tu;  // packed tiles of the kv head
    static uint32_t tile(uint32_t u, uint32_t h) { return (u / k) * Tu + h / Ht; }
    static uint32_t row(uint32_t u, uint32_t h, uint32_t m) { return ((h % Ht) * k + u % k) * M + m; }
};
// one Q tile per unit: a full user whose rows span two packed tiles (M = 8) has units per half (its "kv head" index in
// the schedule is the half; both halves read kv head 0's cache)
constexpr uint32_t kHalvesF = Pack<true>::Tu;
static_assert(Pack<false>::Tu == 1, "a sliding user's rows fit one packed tile");
constexpr uint32_t kPTs = kDts + 2, kPTf = kDtf + 2, kPT = kPTf;  // partial pages: O Dt, m, l
constexpr uint32_t kCS = NVs > NVf ? NVs : NVf;                   // cb_s / cb_p pages

FORCE_INLINE bool is_full(uint32_t l) { return !kDraft && l % 6 == 5; }

// unit-core semaphores (program semaphores: zero every launch). A root's partials count on its own semaphore (they
// arrive in any order): sliding roots U_PARTS + 0..2, the full root U_PARTS + 3. Error words: the BRISC's at the
// engine's attention ids (A_ERR 6, A_ERR2 7, A_ERRU 9), the NCRISC's at the same offsets 4 semaphores on (10, 11, 13:
// F4, one writer per word; the hub reads 6 .. 13 and takes both sets). U_OGADDR: head 0 publishes cb_og's address.
constexpr uint32_t U_READY = 0, U_PARTS = 1, U_SCHED = 5, U_ERR_B = 6, U_ERR2_B = 7, U_OGADDR = 8, U_ERRU_B = 9,
                   U_ERR_N = 10, U_ERR2_N = 11, U_ERRU_N = 13;
// (d_k2a) semaphore 12 (in the hub's error-word read, unused by K2a): 1 = this core's schedule did not fit its records
// (the unit reader writes it every launch; d_hub_dm_v5.cpp counts the cores; never, per GI0)
constexpr uint32_t U_MISS_N = 12;
// head-core semaphores: A_RX 0 and A_OADDR 5 as the streamers address the engine's leaders; O rows in
constexpr uint32_t H_RX = 0, H_OG = 1, H_OADDR = 5;

// status flags (M1_INTERFACE.md s5)
constexpr uint32_t F_POSITION = 0x1, F_SLOT = 0x2, F_TABLE_COLUMN = 0x4, F_PAGE = 0x8;
constexpr uint32_t kGroupsMax = 6;
constexpr uint32_t kGranule = 16;  // table entries per 64 B DRAM read
// the feed rows (positions, slots) in cb_sched, past the builder's scratch (the unit reader reads them at launch)
constexpr uint32_t kFeedPos = 768, kFeedSlot = 800;

constexpr bool group_full(uint32_t g) { return (kGroupKinds >> (8 + g)) & 1; }
constexpr bool group_used(uint32_t g) { return ((kGroupKinds >> g) | (kGroupKinds >> (8 + g))) & 1; }

// ---- unit-core CBs (compute-visible ones < 32)
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_one = 4, cb_s = 5, cb_p = 6, cb_m = 7, cb_l = 8,
                   cb_own = 9, cb_parts = 10, cb_out = 11, cb_sc = 12, cb_diag = 13, cb_cp = 14, cb_ra = 15,
                   cb_rb = 16, cb_sched = 17;
constexpr uint32_t cb_newk = 24, cb_newv = 25, cb_tab = 26, cb_ts = 27;  // data-movement scratch
constexpr uint32_t kMaskPages = 4;
// ---- head-core CBs
constexpr uint32_t hcb_rx = 0, hcb_w = 1, hcb_cs = 2, hcb_one = 3, hcb_sq = 4, hcb_rinv = 5, hcb_xn = 6, hcb_qo = 7,
                   hcb_kr = 8, hcb_vn = 9, hcb_kv8 = 10, hcb_og = 11, hcb_feed = 24;
constexpr uint32_t kRxQ = 0, kRxKV = 16, kRx = 32;  // rx pages: q tiles [0, Dt), k / v (kv) tiles [16, 16 + Dt)
constexpr uint32_t kCSPages = 2 * kDts + 2 * kDtf;  // cos_s [0, 8), sin_s [8, 16), cos_f [16, 32), sin_f [32, 48)


#ifndef COMPILE_FOR_TRISC
FORCE_INLINE uint32_t cb_base(uint32_t cb) {
    auto& iface = get_local_cb_interface(cb);
    return iface.fifo_limit - iface.fifo_size;
}
FORCE_INLINE void copy_words(uint32_t dst, uint32_t src, uint32_t words) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t i = 0; i < words; i++) {
        d[i] = s[i];
    }
}
// rows [rs, rs + n) of the BF8 tile at src into rows [rd, rd + n) of the BF8 tile at dst (each run inside one face-row
// half: the engine's splice_rows with the source rows anywhere)
FORCE_INLINE void splice_rows(uint32_t dst, uint32_t rd, uint32_t src, uint32_t rs, uint32_t n) {
    volatile tt_l1_ptr uint8_t* de = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(dst);
    volatile tt_l1_ptr uint8_t* se = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(src);
    const uint32_t hd = (rd / 16) * 2, od = rd % 16, hs = (rs / 16) * 2, os = rs % 16;
    for (uint32_t f = 0; f < 2; f++) {
        for (uint32_t i = 0; i < n; i++) {
            de[(hd + f) * 16 + od + i] = se[(hs + f) * 16 + os + i];
        }
        copy_words(dst + 64 + (hd + f) * 256 + od * 16, src + 64 + (hs + f) * 256 + os * 16, 4 * n);
    }
}
// user u's M rows of a staged BF8 tile (DRAM page `page`) into the same offsets of the tile at dst: the exponent block
// and, per face of the rows' half, the 64 B-aligned span of their mantissas
FORCE_INLINE void read_rows(const InterleavedAddrGen<true>& gen, uint32_t page, uint32_t u, uint32_t dst) {
    noc_async_read(gen.get_noc_addr(page, 0), dst, 64);
    const uint32_t r = u * M, fh = (r / 16) * 2;
    for (uint32_t f = 0; f < 2; f++) {
        const uint32_t off = 64 + (fh + f) * 256 + (r % 16) * 16;
        const uint32_t a = off & ~63u, e = (off + 16 * M + 63) & ~63u;
        noc_async_read(gen.get_noc_addr(page, a), dst + a, e - a);
    }
}
// splice user u's new rows (positions p .. p + M - 1, staged rows u M ..) into tail tile c (position tile tt0 + c) at
// dst, from the staged tile at src
FORCE_INLINE void splice_user(uint32_t dst, uint32_t src, uint32_t u, uint32_t p, uint32_t tt0, uint32_t c) {
    for (uint32_t m = 0; m < M;) {
        const uint32_t pos = p + m, rd = pos % 32;
        const uint32_t room = 16 - rd % 16, n = M - m < room ? M - m : room;
        if (pos / 32 == tt0 + c) {
            splice_rows(dst, rd, src, u * M + m, n);
        }
        m += n;
    }
}
// mask tile of position tile t for unit user uu (its packed rows: ((h % Ht) k + u % k) M + m) at p: row r of the user
// sees positions p + m - 1023 (sliding) .. p + m; other users' rows see everything (their scores are dropped).
// Drafter: every row of the user sees [p - kDraftW, p + M - 1]
template <bool full>
FORCE_INLINE void build_mask(uint32_t addr, uint32_t t, uint32_t uu, uint32_t p) {
    using P = Pack<full>;
    volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    for (uint32_t face = 0; face < 4; face++) {
        for (uint32_t ri = 0; ri < 16; ri++) {
            const uint32_t r = (face / 2) * 16 + ri, m = r % M, ul = (r / M) % P::k;
            const bool mine = ul == uu % P::k;
            for (uint32_t ci = 0; ci < 16; ci += 2) {
                uint32_t word = 0;
                for (uint32_t e = 0; e < 2; e++) {
                    const uint32_t pos = 32 * t + (face % 2) * 16 + ci + e;
                    const bool hidden =
                        mine && (kDraft ? (pos >= p + M || pos + kDraftW < p)
                                        : (pos > p + m || (!full && pos + (kWindow - 1) < p + m)));
                    word |= static_cast<uint32_t>(hidden ? kNegBig : 0) << (16 * e);
                }
                w[(face * 256 + ri * 16 + ci) / 2] = word;
            }
        }
    }
}
// full layers counted up to and including layer l, and sliding ones
FORCE_INLINE uint32_t layers_of_kind(bool full, uint32_t l) {
    return kDraft ? (full ? 0 : l + 1) : (full ? (l + 1) / 6 : l + 1 - (l + 1) / 6);
}
// (drafter) tile id of (slot, program layer l, kv head g, position tile t, d = 0) in a DrafterRing K or V tensor
// [slots x kRingLayers x 4 x kRingTiles x 32, Dt * 32] (BF8 TILE)
FORCE_INLINE uint32_t ring_tile0(uint32_t slot, uint32_t l, uint32_t g, uint32_t t) {
    return (((slot * kRingLayers + kRingL0 + l) * 4 + g) * kRingTiles + t % kRingTiles) * kDts;
}
#endif

// ---- the launch's users (the hub's validation, M5 launch_of): user u's rows [u M, (u + 1) M), position and slot from
// row u M; inert (-1) or flagged users have no units
struct User {
    uint32_t p, slot, flags;
    bool live;
};
FORCE_INLINE User user_of(volatile tt_l1_ptr uint32_t* pos, volatile tt_l1_ptr uint32_t* slot, uint32_t capacity,
                          uint32_t max_users, uint32_t u) {
    User L{0, 0, 0, false};
    const int32_t p0 = static_cast<int32_t>(pos[u * M]);
    const uint32_t s = slot[u * M];
    if (p0 < 0) {
        return L;
    }
    if (s >= max_users) {
        L.flags = F_SLOT;
        return L;
    }
    if (static_cast<uint32_t>(p0) + M > capacity) {
        L.flags = F_POSITION;
        return L;
    }
    L.p = static_cast<uint32_t>(p0);
    L.slot = s;
    L.live = true;
    return L;
}

// ---- the schedule. Records (words) in cb_sched, the same layout on every RISC of a unit core:
//   [0] units, [1] roots (this core's, both kinds), [2] sliding units, [3] full units (units: sliding first, then full;
//   in each kind non-tail units first, the tail (root) units last), [4] sliding roots (roots: sliding first),
//   [5] Q slots of sliding layers, [6] of full layers, [7] parts pages (the larger kind's), [8] 1 = the build fit,
//   unit i at kUnitRec + kUW i; root j at kRootRec + kRW j; Q slot q at kQRec + q (sliding then full: the packed tile
//   tile: sliding (u / k) | kv head << 8, full (u / k) Tu + half)
constexpr uint32_t kMaxUnits = 20, kMaxRoots = 4, kMaxQ = 4;
// roots per core per kind (sliding: 32 per kv head on 15 cores; drafter: <= 16)
constexpr uint32_t kRootCapS = kDraft ? 2 : 3, kRootCapF = 1;
constexpr uint32_t kUW = 8, kRW = 4;
constexpr uint32_t kUnitRec = 16, kRootRec = kUnitRec + kUW * kMaxUnits, kQRec = kRootRec + kRW * kMaxRoots;
constexpr uint32_t kSchedWords = kQRec + 2 * kMaxQ;
// the builder's scratch (words after the records, in the same cb_sched page): per core load / roots / parts, the
// kind's tail units staged, the users
constexpr uint32_t kScrLoad = 256, kScrRoots = kScrLoad + 64, kScrParts = kScrRoots + 64, kScrTails = kScrParts + 64,
                   kScrUsers = kScrTails + kUW * 8, kScrEnd = kScrUsers + 32;
static_assert(kSchedWords <= kScrLoad && kNumUnits <= 64 && kScrEnd <= 1024, "cb_sched: one 4 KB page");
// unit record: [0] user | g << 8 | full << 12 | tail << 13 | nW << 16 | nT << 20 | qslot << 24
//              [1] t0 (first position tile) | n << 16 (position tiles)
//              [2] root core | parts slot << 8 (this unit's slot among its root's partials, by absolute chunk) |
//                  the root's semaphore index << 16 (U_PARTS + it on the root core)
//              [3] the root's parts base (pages in the root core's cb_parts)
//              [4] (tail units) ct | tt0 << 8 (the tail tiles); [5] (tail units) its root record
// root record: [0] user | g << 8 | full << 12 | k << 16 (partials) | unit index << 24 (its tail unit's record)
//              [1] parts base (pages); [2] its semaphore index (U_PARTS + it)
struct UnitRec {
    volatile tt_l1_ptr uint32_t* w;
    uint32_t user() const { return w[0] & 0xff; }
    uint32_t g() const { return (w[0] >> 8) & 0xf; }
    bool full() const { return (w[0] >> 12) & 1; }
    bool tail() const { return (w[0] >> 13) & 1; }
    uint32_t nW() const { return (w[0] >> 16) & 0xf; }
    uint32_t nT() const { return (w[0] >> 20) & 0xf; }
    uint32_t qslot() const { return w[0] >> 24; }
    uint32_t t0() const { return w[1] & 0xffff; }
    uint32_t n() const { return w[1] >> 16; }
    uint32_t root() const { return w[2] & 0xff; }
    uint32_t pslot() const { return (w[2] >> 8) & 0xff; }
    uint32_t psem() const { return w[2] >> 16; }
    uint32_t pbase() const { return w[3]; }
    uint32_t ct() const { return w[4] & 0xff; }
    uint32_t tt0() const { return w[4] >> 8; }
};
struct RootRec {
    volatile tt_l1_ptr uint32_t* w;
    uint32_t user() const { return w[0] & 0xff; }
    uint32_t g() const { return (w[0] >> 8) & 0xf; }
    bool full() const { return (w[0] >> 12) & 1; }
    uint32_t k() const { return (w[0] >> 16) & 0xff; }
    uint32_t unit() const { return w[0] >> 24; }
    uint32_t pbase() const { return w[1]; }
    uint32_t psem() const { return w[2]; }
};

// sub-chunks of a unit (position tiles [t0, t0 + n)): sliding one; full on the absolute grid of NVf
FORCE_INLINE uint32_t subchunks(bool full, uint32_t t0, uint32_t n) {
    return full ? (t0 + n - 1) / NVf - t0 / NVf + 1 : 1;
}
FORCE_INLINE void subchunk(bool full, uint32_t t0, uint32_t n, uint32_t j, uint32_t& s0, uint32_t& sn) {
    if (!full) {
        s0 = t0;
        sn = n;
        return;
    }
    const uint32_t a = (t0 / NVf + j) * NVf;
    s0 = a > t0 ? a : t0;
    const uint32_t e = a + NVf < t0 + n ? a + NVf : t0 + n;
    sn = e - s0;
}
// local tile t (of n) of a unit is masked: the first nW tiles (window start), the last nT (the tail)
FORCE_INLINE bool masked(uint32_t nW, uint32_t nT, uint32_t n, uint32_t t) { return t < nW || t + nT >= n; }


#ifndef COMPILE_FOR_TRISC
// ---- per-unit page tables (M5 per-unit reads). cb_tab: one kTabW-word region per unit: the user's slot's scratch
// pages per group at [0, 6) (one 64 B scratch-table page read into [0, 16)), its columns lo, n at [8, 10) (written after
// that read), the entries of each used group of the unit's kind from 16 (kColsS / kColsF per group, in group order).
constexpr uint32_t kColsS = 3, kColsF = 17;  // a sliding unit spans <= 9 position tiles, a full one <= 65
constexpr uint32_t kTabW = 64;
FORCE_INLINE uint32_t kind_rank(uint32_t g) {
    uint32_t r = 0;
    for (uint32_t h = 0; h < g; h++) {
        r += (group_used(h) && group_full(h) == group_full(g)) ? 1 : 0;
    }
    return r;
}
struct UnitTab {
    volatile tt_l1_ptr uint32_t* w;  // the unit's region
    uint32_t P;                      // table columns
    uint32_t gmap_arg, pool_arg;     // common runtime arg indices: 8 packed group words, kGroupsMax pool page counts
    uint32_t user;
    uint32_t e0, e1, eu;  // this RISC's error words (semaphore ids)
    uint32_t group(uint32_t l) const { return (get_common_arg_val<uint32_t>(gmap_arg + l / 8) >> (4 * (l % 8))) & 0xf; }
    // physical page of position tile t in layer l (bounds checked; errors -> the scratch page, flagged). Not inlined,
    // called once per position tile (kernel-config text)
    __attribute__((noinline)) uint32_t page(uint32_t l, uint32_t t) const {
        const uint32_t g = group(l);
        const uint32_t col = t / 4, i = col - w[8];
        if (col >= P || i >= w[9]) {
            record(F_TABLE_COLUMN, col, 0);
            return w[g];
        }
        const uint32_t e = w[16 + kind_rank(g) * (group_full(g) ? kColsF : kColsS) + i];
        if (e >= get_common_arg_val<uint32_t>(pool_arg + g)) {
            record(F_PAGE, col, e);
            return w[g];
        }
        return e;
    }
    // tile id of (position tile t, head, d = 0) in a [pages, H, 128, D] BF8 cache
    uint32_t kv_tile0(uint32_t l, bool full, uint32_t t, uint32_t head) const {
        const uint32_t H = full ? 1 : 4, Dt = full ? kDtf : kDts;
        return ((page(l, t) * H + head) * 4 + t % 4) * Dt;
    }
    __attribute__((noinline)) void record(uint32_t flag, uint32_t col, uint32_t entry) const {
        volatile tt_l1_ptr uint32_t* a = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(e0));
        volatile tt_l1_ptr uint32_t* b = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(e1));
        volatile tt_l1_ptr uint32_t* c = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(eu));
        if ((*a & 0xf) == 0) {
            *a = flag | ((user * M) << 4) | ((col < 4095 ? col : 4095) << 9);  // bits 4..8: the user's first row
            *b = entry;
        } else {
            *a = *a | flag;
        }
        *c = *c | (1u << user);
    }
};
#endif

#ifndef COMPILE_FOR_TRISC
// The greedy schedule, identical on every unit core (k2a.py Schedule replays it). Per kind (sliding kv heads 0..3 on
// their kPerHead cores, then full on all cores), per (g, u) in order: the tail unit first (it picks the root: the
// least loaded core with room for a root of the kind), then the chunks in order (the least loaded core with room for
// a unit); loads are position tiles, + 4 for a root; ties go to the lowest index. Writes this core's records.
// users: live (bit 31) | p, per user, in L1.
struct Builder {
    volatile tt_l1_ptr uint32_t* rec;  // cb_sched (records + scratch)
    uint32_t me;

    // window-masked tiles of (p, kind): the tiles holding p + m - 1023 for some row m (sliding only); drafter: the
    // tile holding p - kDraftW when it starts inside a tile
    static void window(bool full, uint32_t p, uint32_t& tlo, uint32_t& wl) {
        if constexpr (kDraft) {
            tlo = p > kDraftW ? (p - kDraftW) / 32 : 0;
            wl = (p > kDraftW && (p - kDraftW) % 32 != 0) ? 1 : 0;
            return;
        }
        tlo = full ? 0 : (p >= kWindow - 1 ? (p - (kWindow - 1)) / 32 : 0);
        wl = (!full && p + M - 1 >= kWindow - 1) ? (p + M - 1 - (kWindow - 1)) / 32 - tlo + 1 : 0;
    }

    bool build() {
        volatile tt_l1_ptr uint32_t* load = rec + kScrLoad;
        volatile tt_l1_ptr uint32_t* nroot = rec + kScrRoots;
        volatile tt_l1_ptr uint32_t* parts = rec + kScrParts;
        volatile tt_l1_ptr uint32_t* tails = rec + kScrTails;
        volatile tt_l1_ptr uint32_t* users = rec + kScrUsers;
        for (uint32_t c = 0; c < kNumUnits; c++) {
            load[c] = 0;
        }
        uint32_t nu = 0, nr = 0, pmax = 0, nq = 0;
        bool ok = true;
        for (uint32_t kind = 0; kind < 2; kind++) {
            const bool full = kind == 1;
            for (uint32_t c = 0; c < kNumUnits; c++) {
                nroot[c] = 0;
                parts[c] = 0;
            }
            uint32_t ntail = 0;
            const uint32_t heads = full ? kHalvesF : 4, ch = full ? kChunkF : kChunkS;
            const uint32_t PT = full ? kPTf : kPTs, cap = full ? kRootCapF : kRootCapS;
            const uint32_t first_unit = nu;
            for (uint32_t g = 0; g < heads; g++) {
                const uint32_t c0 = full ? 0 : g * kPerHead, cn = full ? kNumUnits : kPerHead;
                for (uint32_t u = 0; u < U; u++) {  // (full: g is the half)
                    const uint32_t uw = users[u];
                    if (!(uw >> 31)) {
                        continue;
                    }
                    const uint32_t p = uw & 0x7fffffff;
                    const uint32_t tt0 = p / 32, tend = (p + M - 1) / 32;
                    uint32_t tlo, wl;
                    window(full, p, tlo, wl);
                    const uint32_t clo = tlo / ch, ctl = tt0 / ch, k = ctl - clo + 1;
                    if ((full && k > kMaxChunksF) || (!full && k > kMaxKS)) {
                        ok = false;
                        continue;
                    }
                    uint32_t root = ~0u;
                    for (uint32_t c = c0; c < c0 + cn; c++) {
                        if (nroot[c] < cap && (root == ~0u || load[c] < load[root])) {
                            root = c;
                        }
                    }
                    if (root == ~0u) {
                        ok = false;
                        continue;
                    }
                    const uint32_t pbase = parts[root];
                    const uint32_t psem = (full ? kRootCapS : 0) + nroot[root];
                    parts[root] += k * PT;
                    nroot[root]++;
                    const uint32_t a0 = tlo > ctl * ch ? tlo : ctl * ch;
                    load[root] += tend - a0 + 1 + 4;
                    const uint32_t wend = tlo + wl;  // window-masked tiles [tlo, wend)
                    if (root == me) {
                        if (nr >= kMaxRoots || ntail >= 8) {
                            ok = false;
                        } else {
                            volatile tt_l1_ptr uint32_t* r = rec + kRootRec + kRW * nr;
                            r[0] = u | g << 8 | (full ? 1u : 0u) << 12 | k << 16;
                            r[1] = pbase;
                            r[2] = psem;
                            volatile tt_l1_ptr uint32_t* t = tails + kUW * ntail;
                            const uint32_t n = tend - a0 + 1;
                            const uint32_t nw = a0 < wend ? (wend - a0 < n ? wend - a0 : n) : 0;
                            t[0] = u | g << 8 | (full ? 1u : 0u) << 12 | 1u << 13 | nw << 16 | (tend - tt0 + 1) << 20;
                            t[1] = a0 | n << 16;
                            t[2] = root | (ctl - clo) << 8 | psem << 16;
                            t[3] = pbase;
                            t[4] = (tend - tt0 + 1) | tt0 << 8;
                            t[5] = nr;
                            ntail++;
                            nr++;
                        }
                    }
                    for (uint32_t cc = clo; cc < ctl; cc++) {
                        uint32_t b = ~0u;
                        for (uint32_t c = c0; c < c0 + cn; c++) {
                            if (b == ~0u || load[c] < load[b]) {
                                b = c;
                            }
                        }
                        const uint32_t s0 = tlo > cc * ch ? tlo : cc * ch;
                        const uint32_t n = (cc + 1) * ch - s0;
                        load[b] += n;
                        if (b == me) {
                            if (nu - first_unit + ntail >= kMaxUnits || nu >= kMaxUnits) {
                                ok = false;
                                continue;
                            }
                            volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * nu;
                            const uint32_t nw = s0 < wend ? (wend - s0 < n ? wend - s0 : n) : 0;
                            w[0] = u | g << 8 | (full ? 1u : 0u) << 12 | nw << 16;
                            w[1] = s0 | n << 16;
                            w[2] = root | (cc - clo) << 8 | psem << 16;
                            w[3] = pbase;
                            w[4] = 0;
                            w[5] = 0;
                            nu++;
                        }
                    }
                }
            }
            // the kind's tail units after its other units; their roots point at them
            for (uint32_t i = 0; i < ntail; i++) {
                if (nu >= kMaxUnits) {
                    ok = false;
                    break;
                }
                volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * nu;
                for (uint32_t x = 0; x < 6; x++) {
                    w[x] = tails[kUW * i + x];
                }
                volatile tt_l1_ptr uint32_t* r = rec + kRootRec + kRW * tails[kUW * i + 5];
                r[0] = r[0] | nu << 24;
                nu++;
            }
            // Q slots: the packed tile groups this kind's units read (sliding: per kv head)
            const uint32_t kq = full ? Pack<true>::k : Pack<false>::k;
            const uint32_t q0 = nq;
            for (uint32_t i = first_unit; i < nu; i++) {
                volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * i;
                // the packed tile within the kv head's set (full: the user group's half) | sliding kv head << 8
                const uint32_t uu = w[0] & 0xff, gg = (w[0] >> 8) & 0xf;
                const uint32_t key = full ? (uu / kq) * kHalvesF + gg : (uu / kq) | gg << 8;
                uint32_t q = q0;
                while (q < nq && rec[kQRec + q] != key) {
                    q++;
                }
                if (q == nq) {
                    if (nq - q0 >= kMaxQ) {
                        ok = false;
                        break;
                    }
                    rec[kQRec + nq++] = key;
                }
                w[0] = (w[0] & 0x00ffffff) | (q - q0) << 24;
            }
            if (parts[me] > pmax) {
                pmax = parts[me];
            }
            if (kind == 0) {
                rec[2] = nu;
                rec[4] = nr;
                rec[5] = nq;
            } else {
                rec[6] = nq - rec[5];
            }
        }
        rec[0] = nu;
        rec[1] = nr;
        rec[3] = nu - rec[2];
        rec[7] = pmax;
        ok = ok && pmax <= NP;
        rec[8] = ok ? 1 : 0;
        return ok;
    }
};
#endif

}  // namespace k2a
