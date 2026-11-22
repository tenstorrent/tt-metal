// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention at R = 32 for U users of M rows (M5 K2a, bench/resident/m5/K2A_DESIGN_v2.md + addenda 1-2):
// 8 head cores (one per q head) and 60 unit cores per chip.
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
constexpr uint32_t CQ = get_compile_time_arg_val(12);          // cb_q pages: (K2b step 1) a ring of 2 Q slots of kQSlot

// (M5 D16) the launch rows: 32, or 8 (multi-user R = 8: the head cores pad the streamers' 8-row tiles to 32 rows,
// so everything past the head readers sees 32-row tiles as at R = 32)
constexpr uint32_t R = get_compile_time_arg_val(13);
static_assert(R == 32 || (R == 8 && M < 8), "launch rows");
constexpr uint32_t U = R / M;  // users per launch
static_assert(M == 1 || M == 2 || M == 4 || M == 8, "rows per user");
constexpr uint32_t kHeads = 8;  // q heads per chip (= head cores)
constexpr uint32_t kDts = 8, kDtf = 16;
constexpr uint32_t kBf8 = 1088, kBf16 = 2048;
constexpr uint32_t kPage = 128, kWindow = 1024;
constexpr uint32_t kChunkS = 8, kChunkF = 64;  // absolute chunks (position tiles)
constexpr uint32_t NVs = 8, NVf = 8;           // sub-chunk tiles: sliding = the whole unit (<= 8: m5_14 F1, no
                                               // straddling tail tile), full = the absolute grid of 8
constexpr uint32_t kMaxChunksF = 8;            // full units per user: positions < 8 * 2048 (K2b step 1: the flat merge
                                               // of <= 8 partials fits cb_parts (8 x 18 <= NP) and cb_sc / cb_diag (8))
constexpr uint16_t kNegBig = 0xce6e;           // bf16 -1e9 (as the engine's attention)

// packed Q layout of kv head g (QH q heads): user u's QH * M rows over T_u tiles of H_t heads, k users per tile
template <bool full>
struct Pack {
    static constexpr uint32_t QH = full ? 8 : 2;
    static constexpr uint32_t Ru = QH * M;
    static constexpr uint32_t Tu = (Ru + 31) / 32;   // Q tiles per user (2: full at M = 8)
    static constexpr uint32_t Ht = QH / Tu;          // heads per tile
    static constexpr uint32_t k = 32 / (Ht * M);     // users per tile
    static constexpr uint32_t groups = (U + k - 1) / k;  // user groups (D16: a part-filled last group at R = 8)
    static constexpr uint32_t tiles = groups * Tu;         // packed tiles of the kv head
    static uint32_t tile(uint32_t u, uint32_t h) { return (u / k) * Tu + h / Ht; }
    static uint32_t row(uint32_t u, uint32_t h, uint32_t m) { return ((h % Ht) * k + u % k) * M + m; }
};
// one Q tile per unit: a full user whose rows span two packed tiles (M = 8) has units per half (its "kv head" index in
// the schedule is the half; both halves read kv head 0's cache)
constexpr uint32_t kHalvesF = Pack<true>::Tu;
static_assert(Pack<false>::Tu == 1, "a sliding user's rows fit one packed tile");
constexpr uint32_t kPTs = kDts + 2, kPTf = kDtf + 2, kPT = kPTf;  // partial pages: O Dt, m, l
constexpr uint32_t kCS = NVs > NVf ? NVs : NVf;                   // cb_s / cb_p pages

FORCE_INLINE bool is_full(uint32_t l) { return l % 6 == 5; }

// unit-core semaphores (program semaphores: zero every launch). A root's partials count on its own semaphore (they
// arrive in any order): sliding roots U_PARTS + 0..2, the full root U_PARTS + 3. Error words: the BRISC's at the
// engine's attention ids (A_ERR 6, A_ERR2 7, A_ERRU 9), the NCRISC's at the same offsets 4 semaphores on (10, 11, 13:
// F4, one writer per word; the hub reads 6 .. 13 and takes both sets). U_OGADDR: head 0 publishes cb_og's address.
constexpr uint32_t U_READY = 0, U_PARTS = 1, U_SCHED = 5, U_ERR_B = 6, U_ERR2_B = 7, U_OGADDR = 8, U_ERRU_B = 9,
                   U_ERR_N = 10, U_ERR2_N = 11, U_ERRU_N = 13;
// (K2b step 3) U_SCHERR: unit core 0's schedule-overflow word (the hub reads ids 6 .. 13: F_SCHED); U_LANES, + 1:
// the group nodes' input counters (4 byte lanes each). Head cores: H_SCHED, the schedule status from unit core 0
// (1 fit, 2 overflow: every row inert), waited on once per launch
constexpr uint32_t U_SCHERR = 12, U_LANES = 14, H_SCHED = 6;
constexpr uint32_t kSchedOk = 1, kSchedOverflow = 2;
// head-core semaphores: A_RX 0 and A_OADDR 5 as the streamers address the engine's leaders; O rows in
constexpr uint32_t H_RX = 0, H_OG = 1, H_OADDR = 5;

// status flags (M1_INTERFACE.md s5)
constexpr uint32_t F_POSITION = 0x1, F_SLOT = 0x2, F_TABLE_COLUMN = 0x4, F_PAGE = 0x8;
constexpr uint32_t kGroupsMax = 6;
constexpr uint32_t kGranule = 16;  // table entries per 64 B DRAM read

constexpr bool group_full(uint32_t g) { return (kGroupKinds >> (8 + g)) & 1; }
constexpr bool group_used(uint32_t g) { return ((kGroupKinds >> g) | (kGroupKinds >> (8 + g))) & 1; }

// ---- unit-core CBs (compute-visible ones < 32)
constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_mask = 3, cb_one = 4, cb_s = 5, cb_p = 6, cb_m = 7, cb_l = 8,
                   cb_own = 9, cb_parts = 10, cb_out = 11, cb_sc = 12, cb_diag = 13, cb_cp = 14, cb_ra = 15,
                   cb_rb = 16, cb_sched = 17, cb_gin = 18;  // (K2b step 3) cb_gin: group-node inputs (BRISC -> compute)
constexpr uint32_t cb_newk = 24, cb_newv = 25, cb_tab = 26, cb_ts = 27;  // data-movement scratch
constexpr uint32_t kMaskPages = 4;
// ---- head-core CBs
constexpr uint32_t hcb_rx = 0, hcb_w = 1, hcb_cs = 2, hcb_one = 3, hcb_sq = 4, hcb_rinv = 5, hcb_xn = 6, hcb_qo = 7,
                   hcb_kr = 8, hcb_vn = 9, hcb_kv8 = 10, hcb_og = 11, hcb_feed = 24;
// (M5 D16) R = 8: the streamers' 8-row tiles land in hcb_rx8 (32 x 64 R B); the head reader pads them into hcb_rx's
// 32-row tiles; O goes back as 8-row tiles through hcb_o8 (16 x 64 R B)
constexpr uint32_t hcb_rx8 = 12, hcb_o8 = 13;
constexpr uint32_t kTinyBytes = 64 * R;  // a streamer tile (R rows x 32)
constexpr uint32_t kRxQ = 0, kRxKV = 16, kRx = 32;  // rx pages: q tiles [0, Dt), k / v (kv) tiles [16, 16 + Dt)
constexpr uint32_t kCSPages = 2 * kDts + 2 * kDtf;  // cos_s [0, 8), sin_s [8, 16), cos_f [16, 32), sin_f [32, 48)


#ifndef COMPILE_FOR_TRISC
// (CP1 timing, m5_3_1) K2A_TL = the L1 address of this core's timeline words (k2a.py, timeline on); without the define
// the marks compile out and the binaries are m5_3's
#ifdef K2A_TL
#define K2A_MARK(l, i, words)                                                                      \
    (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(K2A_TL)[(l) * (words) + (i)] =                  \
         *reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L))
#else
#define K2A_MARK(l, i, words) ((void)0)
#endif
// timeline words per layer: head [rx, READY, O in, sem_o, (D16) pad done, repack done]; unit [Q, O]
constexpr uint32_t kTlHead = 6, kTlUnit = 2;
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
// sees positions p + m - 1023 (sliding) .. p + m; other users' rows see everything (their scores are dropped)
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
                    const bool hidden = mine && (pos > p + m || (!full && pos + (kWindow - 1) < p + m));
                    word |= static_cast<uint32_t>(hidden ? kNegBig : 0) << (16 * e);
                }
                w[(face * 256 + ri * 16 + ci) / 2] = word;
            }
        }
    }
}
// full layers counted up to and including layer l, and sliding ones
FORCE_INLINE uint32_t layers_of_kind(bool full, uint32_t l) { return full ? (l + 1) / 6 : l + 1 - (l + 1) / 6; }
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
//   [5] Q keys of sliding layers, [6] of full layers, [7] parts pages (the larger kind's), [8] 1 = the build fit
//   (K2b step 3: global, the same on every unit core), [9] group nodes (full layers, node n at kNodeRec + kNW n),
//   unit i at kUnitRec + kUW i; root j at kRootRec + kRW j; Q key q at kQRec + q (sliding then full: the packed tile
//   tile: sliding (u / k) | kv head << 8, full (u / k) Tu + half)
// (K2b step 1) 40 units (pool-bounded mixes reach 27); kMaxQ bounds the per-kind key list only: Q is streamed one key
// (packed tile) at a time through a 2-slot ring of kQSlot pages, in the key order the units are sorted by
constexpr uint32_t kMaxUnits = 40, kMaxRoots = 4, kMaxQ = 8;
constexpr uint32_t kQSlot = kDtf;  // pages per Q push (sliding keys use the first kDts; no ring straddle)
constexpr uint32_t kRootCapS = 3, kRootCapF = 1;  // roots per core per kind (sliding: 32 per kv head on 15 cores)
constexpr uint32_t kUW = 8, kRW = 4, kNW = 4;
// (K2b step 3) long users (full layers, k > kMaxChunksF chunks, up to kMaxChunksT): every chunk partial goes to the
// DRAM partials buffer [user, half, chunk, kPTf pages]; a group node per kGroup absolute chunks [kGroup j, kGroup (j + 1))
// folds its inputs (merge2, unnormalized) into the root's parts slot j; the root merges the <= kMaxChunksT / kGroup
// group partials (merge_root). A node counts its inputs on a byte lane of semaphore U_LANES + lane / 4 (4 lanes per
// word, kMaxNodes per core). The unit record's [0] bit 14 marks a long user's unit: its [2] is node core | absolute
// chunk << 8 | (kLaneTag + lane) << 16 and [3] is unused.
constexpr uint32_t kMaxChunksT = 128, kGroup = 16, kMaxNodes = 8, kLaneTag = 16;
static_assert(kMaxChunksT / kGroup <= kMaxChunksF, "the root merges <= kMaxChunksF group partials (cb_parts, cb_sc)");
constexpr uint32_t kUnitRec = 16, kRootRec = kUnitRec + kUW * kMaxUnits, kQRec = kRootRec + kRW * kMaxRoots;
constexpr uint32_t kNodeRec = kQRec + 2 * kMaxQ, kSchedWords = kNodeRec + kNW * kMaxNodes;
// the builder's scratch (words after the records, in the same cb_sched page): per core load / roots / parts, the
// kind's tail units staged, the users; (K2b step 3, the global fit) per core units, group tasks and tails of the kind
constexpr uint32_t kScrLoad = 416, kScrRoots = kScrLoad + 64, kScrParts = kScrRoots + 64, kScrTails = kScrParts + 64,
                   kScrUsers = kScrTails + kUW * 8, kScrUnits = kScrUsers + 32, kScrTasks = kScrUnits + 64,
                   kScrTailN = kScrTasks + 64, kScrEnd = kScrTailN + 64;
constexpr uint32_t kFeedPos = 896, kFeedSlot = 928;  // the feed rows (32 words each, read by the unit reader)
static_assert(kSchedWords <= kScrLoad && kNumUnits <= 64 && kScrEnd <= kFeedPos && kFeedSlot + 32 <= 1024,
              "cb_sched: one 4 KB page");
// unit record: [0] user | g << 8 | full << 12 | tail << 13 | nW << 16 | nT << 20 | qslot << 24 (K2b step 1: the
//              unit's key index; a kind's units are stably sorted by (tail, key), so the compute pops a key's slot
//              when it changes and the reader pushes one per run of equal keys)
//              [1] t0 (first position tile) | n << 16 (position tiles)
//              [2] root core | parts slot << 8 (this unit's slot among its root's partials, by absolute chunk) |
//                  the root's semaphore index << 16 (U_PARTS + it on the root core)
//              [3] the root's parts base (pages in the root core's cb_parts)
//              [4] (tail units) ct | tt0 << 8 (the tail tiles); [5] (tail units) its root record
//              (m5_14 F1) bit 15 of [0] marks a piece: the chunk unit of a block straddling a chunk boundary that
//              holds the block's first rows. It carries tail (bit 13), nT and [4] like a tail unit (it splices and
//              writes back those rows, masks its last nT tiles, waits for READY) but is not its root's unit ([5] 0)
// node record: [0] user | g << 8 | n inputs << 16 | lane << 24; [1] first absolute chunk; [2] / [3] as a unit's
//              (the root core | parts slot j << 8 | psem << 16; the root's parts base): its output's destination
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
    bool dram() const { return (w[0] >> 14) & 1; }  // (K2b step 3) a long user's unit: partial to DRAM
    bool piece() const { return (w[0] >> 15) & 1; }  // (m5_14 F1) a straddling block's chunk piece
};
struct NodeRec {  // (K2b step 3) a group node
    volatile tt_l1_ptr uint32_t* w;
    uint32_t user() const { return w[0] & 0xff; }
    uint32_t g() const { return (w[0] >> 8) & 0xf; }
    uint32_t n() const { return (w[0] >> 16) & 0xff; }
    uint32_t lane() const { return w[0] >> 24; }
    uint32_t c0() const { return w[1]; }
    uint32_t root() const { return w[2] & 0xff; }
    uint32_t pslot() const { return (w[2] >> 8) & 0xff; }
    uint32_t psem() const { return w[2] >> 16; }
    uint32_t pbase() const { return w[3]; }
};
// (K2b step 3) a long user's partial x (absolute chunk c, half g) in the DRAM partials buffer: kPTf pages
FORCE_INLINE uint32_t dram_part_page(uint32_t user, uint32_t g, uint32_t c) {
    return ((user * kHalvesF + g) * kMaxChunksT + c) * kPTf;
}
// the byte-lane increment a long unit sends to its node's counter (semaphore U_LANES + lane / 4)
FORCE_INLINE uint32_t lane_sem(uint32_t lane) { return U_LANES + lane / 4; }
FORCE_INLINE uint32_t lane_inc(uint32_t lane) { return 1u << (8 * (lane % 4)); }
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
constexpr uint32_t kColsS = 3, kColsF = 17;  // a sliding unit spans <= 9 position tiles, a full one <= 65 (m5_14 F1:
                                             // <= 8 / 64, chunk-aligned; the regions keep m5_13's sizes)
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
// (m5_14 F1) the tail chunk is the block's last tile's (ctl = tend / ch). A block straddling a chunk boundary (its
// first tile tt0 in chunk ctl - 1) leaves those rows' tile to chunk ctl - 1's unit, a piece (bit 15) that splices them,
// so every position tile is reduced in its own absolute chunk whatever the block's alignment (Q2). m5_13 took
// ctl = tt0 / ch and reduced the straddled tile inside chunk ctl - 1's tail (W1PROBE v3: onset at 1280). A sliding user
// can then span 6 chunks (M >= 4, p = 256 c - 2 or 256 c - 3 for M = 4, ..), so a root also needs room for its
// partials (parts + ng PT <= NP), which never binds while every k <= 5 (3 x 5 x kPTs = NP): m5_13's choice there.
// A block straddling 16,384 (full chunk 8) would make k = 9, a long user, so its rows below 16,384 would take the
// group-node tree where an aligned block's take the flat merge: the engine caps such a block there (verify_cap, M131)
// and the rows from 16,384 on come inert, so the block's extent (tend) runs to its last live row.
struct Builder {
    volatile tt_l1_ptr uint32_t* rec;  // cb_sched (records + scratch)
    uint32_t me;

    // window-masked tiles of (p, kind): the tiles holding p + m - 1023 for some row m < ml (sliding only)
    static void window(bool full, uint32_t p, uint32_t ml, uint32_t& tlo, uint32_t& wl) {
        tlo = full ? 0 : (p >= kWindow - 1 ? (p - (kWindow - 1)) / 32 : 0);
        wl = (!full && p + ml - 1 >= kWindow - 1) ? (p + ml - 1 - (kWindow - 1)) / 32 - tlo + 1 : 0;
    }

    bool build() {
        volatile tt_l1_ptr uint32_t* load = rec + kScrLoad;
        volatile tt_l1_ptr uint32_t* nroot = rec + kScrRoots;
        volatile tt_l1_ptr uint32_t* parts = rec + kScrParts;
        volatile tt_l1_ptr uint32_t* tails = rec + kScrTails;
        volatile tt_l1_ptr uint32_t* users = rec + kScrUsers;
        // (K2b step 3) per core, every core: units (both kinds), group tasks, tails of the kind; every fit check below
        // is evaluated for every core, so ok (rec[8]) is the same on all unit cores
        volatile tt_l1_ptr uint32_t* nunits = rec + kScrUnits;
        volatile tt_l1_ptr uint32_t* ntask = rec + kScrTasks;
        volatile tt_l1_ptr uint32_t* ntailc = rec + kScrTailN;
        for (uint32_t c = 0; c < kNumUnits; c++) {
            load[c] = 0;
            nunits[c] = 0;
            ntask[c] = 0;
        }
        uint32_t nu = 0, nr = 0, pmax = 0, nq = 0, nn = 0;
        bool ok = true;
        for (uint32_t kind = 0; kind < 2; kind++) {
            const bool full = kind == 1;
            for (uint32_t c = 0; c < kNumUnits; c++) {
                nroot[c] = 0;
                parts[c] = 0;
                ntailc[c] = 0;
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
                    // (m5_14 F1, M131) the block's rows to its last live one (feed rows >= 0): a block the engine
                    // capped at a merge boundary (verify_cap) ends in inert rows, which take no tiles
                    uint32_t ml = 1;
                    for (uint32_t m = 1; m < M; m++) {
                        if (static_cast<int32_t>(rec[kFeedPos + u * M + m]) >= 0) {
                            ml = m + 1;
                        }
                    }
                    const uint32_t tt0 = p / 32, tend = (p + ml - 1) / 32;
                    uint32_t tlo, wl;
                    window(full, p, ml, tlo, wl);
                    // (m5_14 F1) ctl: the last tile's chunk; cfirst: the first tile's (ctl - 1 when straddling)
                    const uint32_t clo = tlo / ch, ctl = tend / ch, k = ctl - clo + 1, cfirst = tt0 / ch;
                    // (K2b step 3) a long user (full, k > kMaxChunksF): its root merges ng group partials
                    const bool lng = full && k > kMaxChunksF;
                    if (full && k > kMaxChunksT) {
                        ok = false;
                        continue;
                    }
                    const uint32_t ng = lng ? (k + kGroup - 1) / kGroup : k;
                    uint32_t root = ~0u;
                    for (uint32_t c = c0; c < c0 + cn; c++) {
                        if (nroot[c] < cap && parts[c] + ng * PT <= NP && (root == ~0u || load[c] < load[root])) {
                            root = c;
                        }
                    }
                    if (root == ~0u) {
                        ok = false;
                        continue;
                    }
                    const uint32_t pbase = parts[root];
                    const uint32_t psem = (full ? kRootCapS : 0) + nroot[root];
                    parts[root] += ng * PT;
                    nroot[root]++;
                    const uint32_t a0 = tlo > ctl * ch ? tlo : ctl * ch;
                    load[root] += tend - a0 + 1 + 4;
                    const uint32_t wend = tlo + wl;  // window-masked tiles [tlo, wend)
                    // (K2b step 3) the long user's group nodes, placed by fewest group tasks (then load, then index)
                    uint32_t ncore[kMaxChunksT / kGroup] = {}, nlane[kMaxChunksT / kGroup] = {};
                    for (uint32_t j = 0; lng && j < ng; j++) {
                        uint32_t b = 0;
                        for (uint32_t c = 1; c < kNumUnits; c++) {
                            if (ntask[c] < ntask[b] || (ntask[c] == ntask[b] && load[c] < load[b])) {
                                b = c;
                            }
                        }
                        const uint32_t n_j = k - kGroup * j < kGroup ? k - kGroup * j : kGroup;
                        if (ntask[b] >= kMaxNodes) {
                            ok = false;
                        }
                        ncore[j] = b;
                        nlane[j] = ntask[b] % kMaxNodes;
                        ntask[b]++;
                        load[b] += n_j;
                        if (b == me && nn < kMaxNodes) {
                            volatile tt_l1_ptr uint32_t* w = rec + kNodeRec + kNW * nn;
                            w[0] = u | g << 8 | n_j << 16 | nlane[j] << 24;
                            w[1] = kGroup * j;
                            w[2] = root | j << 8 | psem << 16;
                            w[3] = pbase;
                            nn++;
                        }
                    }
                    // a chunk's partial: to the root's slot (short) or to DRAM and its node's lane (long)
                    auto dest = [&](volatile tt_l1_ptr uint32_t* w, uint32_t cc) {
                        if (lng) {
                            w[0] = w[0] | 1u << 14;
                            w[2] = ncore[cc / kGroup] | cc << 8 | (kLaneTag + nlane[cc / kGroup]) << 16;
                            w[3] = 0;
                        } else {
                            w[2] = root | (cc - clo) << 8 | psem << 16;
                            w[3] = pbase;
                        }
                    };
                    nunits[root]++;
                    ntailc[root]++;
                    if (nunits[root] > kMaxUnits || ntailc[root] > 8) {
                        ok = false;
                    }
                    if (root == me && ok) {
                        volatile tt_l1_ptr uint32_t* r = rec + kRootRec + kRW * nr;
                        r[0] = u | g << 8 | (full ? 1u : 0u) << 12 | ng << 16;
                        r[1] = pbase;
                        r[2] = psem;
                        volatile tt_l1_ptr uint32_t* t = tails + kUW * ntail;
                        const uint32_t n = tend - a0 + 1;
                        const uint32_t nw = a0 < wend ? (wend - a0 < n ? wend - a0 : n) : 0;
                        // (m5_14 F1) the block's tiles in this unit: [max(tt0, a0), tend] (all n when straddling)
                        const uint32_t nt = tend - (tt0 > a0 ? tt0 : a0) + 1;
                        t[0] = u | g << 8 | (full ? 1u : 0u) << 12 | 1u << 13 | nw << 16 | nt << 20;
                        t[1] = a0 | n << 16;
                        dest(t, ctl);
                        t[4] = nt | tt0 << 8;
                        t[5] = nr;
                        ntail++;
                        nr++;
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
                        nunits[b]++;
                        if (nunits[b] > kMaxUnits) {
                            ok = false;
                        }
                        if (b == me && ok) {
                            volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * nu;
                            const uint32_t nw = s0 < wend ? (wend - s0 < n ? wend - s0 : n) : 0;
                            w[0] = u | g << 8 | (full ? 1u : 0u) << 12 | nw << 16;
                            w[1] = s0 | n << 16;
                            dest(w, cc);
                            w[4] = 0;
                            w[5] = 0;
                            if (cc == cfirst) {
                                // (m5_14 F1) the straddling block's piece: its tiles [tt0, (cc + 1) ch) as a tail's
                                const uint32_t nt = (cc + 1) * ch - tt0;
                                w[0] = w[0] | 1u << 13 | 1u << 15 | nt << 20;
                                w[4] = nt | tt0 << 8;
                            }
                            nu++;
                        }
                    }
                }
            }
            // the kind's tail units after its other units; their roots point at them
            for (uint32_t i = 0; i < ntail && ok; i++) {
                volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * nu;
                for (uint32_t x = 0; x < 6; x++) {
                    w[x] = tails[kUW * i + x];
                }
                nu++;
            }
            // Q keys: the packed tile groups this kind's units read (sliding: per kv head), in first-use order
            const uint32_t kq = full ? Pack<true>::k : Pack<false>::k;
            const uint32_t q0 = nq;
            for (uint32_t i = first_unit; i < nu && ok; i++) {
                volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * i;
                // the packed tile within the kv head's set (full: the user group's half) | sliding kv head << 8
                const uint32_t uu = w[0] & 0xff, gg = (w[0] >> 8) & 0xf;
                const uint32_t key = full ? (uu / kq) * kHalvesF + gg : (uu / kq) | gg << 8;
                uint32_t q = q0;
                while (q < nq && rec[kQRec + q] != key) {
                    q++;
                }
                if (q == nq) {
                    // (kMaxQ is the true maximum of packed tiles per kind: a local check suffices)
                    if (nq - q0 >= kMaxQ) {
                        ok = false;
                        break;
                    }
                    rec[kQRec + nq++] = key;
                }
                w[0] = (w[0] & 0x00ffffff) | (q - q0) << 24;
            }
            // (K2b step 1) the kind's units stably sorted by (tail, key index) (insertion sort, <= kMaxUnits records):
            // the reader streams one Q slot per run of equal keys, in this order. Only this core's unit order changes (each unit's arithmetic
            // and the roots' merge by absolute chunk do not), so Q2 holds and k <= 5 stays bitwise
            for (uint32_t i = first_unit + 1; i < nu && ok; i++) {
                uint32_t tmp[6];
                volatile tt_l1_ptr uint32_t* wi = rec + kUnitRec + kUW * i;
                for (uint32_t x = 0; x < 6; x++) {
                    tmp[x] = wi[x];
                }
                uint32_t j = i;
                // (K2b: non-tail units first, tails last, each part by key: the K / V prefetch of non-tail units runs
                // ahead of READY, which the tails need; k2b_tl_probe.json: sorting tails in with their keys cost
                // +0.9 ms per step at 32 x 1 @ 640, all in the sliding layers)
                auto order = [](uint32_t w0) { return ((w0 >> 13) & 1) << 8 | (w0 >> 24); };
                while (j > first_unit && order(rec[kUnitRec + kUW * (j - 1)]) > order(tmp[0])) {
                    for (uint32_t x = 0; x < 6; x++) {
                        rec[kUnitRec + kUW * j + x] = rec[kUnitRec + kUW * (j - 1) + x];
                    }
                    j--;
                }
                for (uint32_t x = 0; x < 6; x++) {
                    rec[kUnitRec + kUW * j + x] = tmp[x];
                }
            }
            // the roots point at their tail units' new records ((m5_14 F1) not at pieces)
            for (uint32_t i = first_unit; i < nu && ok; i++) {
                volatile tt_l1_ptr uint32_t* w = rec + kUnitRec + kUW * i;
                if (((w[0] >> 13) & 1) && !((w[0] >> 15) & 1)) {
                    volatile tt_l1_ptr uint32_t* r = rec + kRootRec + kRW * w[5];
                    r[0] = (r[0] & 0x00ffffff) | i << 24;
                }
            }
            // (K2b step 3) parts fit on every core (the global check)
            for (uint32_t c = 0; c < kNumUnits; c++) {
                if (parts[c] > NP) {
                    ok = false;
                }
                if (parts[c] > pmax) {
                    pmax = parts[c];
                }
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
        rec[9] = nn;
        rec[8] = ok ? 1 : 0;
        if (!ok) {
            // (K2b step 3) overflow: the launch runs inert on every unit core (no units, roots, nodes, Q keys)
            for (uint32_t x = 0; x < 7; x++) {
                rec[x] = 0;
            }
            rec[9] = 0;
        }
        return ok;
    }
};
#endif

}  // namespace k2a
