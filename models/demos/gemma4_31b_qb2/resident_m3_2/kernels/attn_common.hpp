// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Gemma resident attention (step 2, one user, M rows at positions p .. p + M - 1): the shared schedule of the
// attention cores. New for this prototype; the leader / worker split over paged KV follows the design of
// changh95/qwen3.8-27b-p300x2 code/qwen36_resident (attn_common.hpp, Apache-2.0), the per-core partial softmax and
// its merge the flash-decode reduction of ttnn sdpa_decode.
//
// 32 attention cores per chip in 4 groups of 8 (group g: leader L_g = index 0, merger G_g = index 1, workers 2..7).
//   sliding layer (kv head g of this chip in group g, head_dim 256, window 1024, ring of 9 pages / 1152 positions):
//     L_g: q|k|v rows from the streamers -> q / k norms, RoPE, v norm; the new K / V rows spliced into the BF8 cache
//          (one read-modify-write per tile); attention over the tail tiles (the ones holding p .. p + M - 1) from its
//          spliced copies; root: merges the partials, O rows of heads 2g, 2g + 1 to every streamer's o_in.
//     G_g + workers (7 cores): complete position tiles [lo, tail) of the window, split in order.
//   full layer (the chip's one kv head, head_dim 512, K and V stored apart): L_0 preps, takes the tail and is the root;
//     the 28 non-leaders split [0, tail) in group order (G_g first); G_g merges its group's partials and sends one to
//     L_0. L_1..3 idle.
// Partials are (O unnormalized, m = row max, l = row sum) per Q tile; a merge of x = own + k children takes
// M = max m_x, s_x = exp(m_x - M), L = sum s_x l_x, O = sum diag(s_x [/ L at the root]) O_x as one matmul.
// Q rows: i = h * R + r (head h of the group, row r; R = tile rows of the step, padded rows r >= M are computed
// and dropped). K / V rows: r. Every row r sees positions p + r - 1023 .. p + r (sliding) / 0 .. p + r (full).
//
// Serving (step 3, M1_INTERFACE.md): the block's position and slot come from the engine's feed tensors (row 0), the
// pages from the slot's per-group vLLM tables (every layer reads its group's row; under KV aliasing the full group's
// entries are already full-view pages), with bounds checked on every read: a launch whose row 0 is inert (position
// -1), or out of range (position / slot: flagged), runs the schedule of p = 0 against the slot's scratch pages, so
// nothing real is read or written; a table column past the table or an entry outside the layer's pool is flagged
// and replaced by the scratch page (status bits in the A_ERR words, collected by the hub).
#pragma once

#include <stdint.h>

namespace gatt {

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t R = get_compile_time_arg_val(1);
constexpr uint32_t M = get_compile_time_arg_val(2);
constexpr uint32_t num_streamers = get_compile_time_arg_val(3);
constexpr uint32_t QTs = get_compile_time_arg_val(4);  // Q tiles (32 rows) of a sliding group: 2 heads x R rows
constexpr uint32_t QTf = get_compile_time_arg_val(5);  // of the full layer: 8 heads x R rows
constexpr uint32_t CQ = get_compile_time_arg_val(6);   // cb_q pages (every layer pushes all of them)
constexpr uint32_t NVs = get_compile_time_arg_val(7);  // V window (tiles per d) of a sliding worker
constexpr uint32_t NVf = get_compile_time_arg_val(8);  // of a full worker: its chunk (M1L: a full worker walks its
                                                       // tiles in chunks of NVf, merging a running partial)
constexpr uint32_t PTs = get_compile_time_arg_val(9);  // partial tiles: QT x Dt O, QT m, QT l
constexpr uint32_t PTf = get_compile_time_arg_val(10);
constexpr uint32_t CV = get_compile_time_arg_val(11);  // cb_v pages (a multiple of NVs and NVf)
constexpr uint32_t CS = get_compile_time_arg_val(12);  // cb_s / cb_p pages
constexpr uint32_t eps_bits = get_compile_time_arg_val(13);
constexpr uint32_t inv_ds_bits = get_compile_time_arg_val(14);  // 1 / 256
constexpr uint32_t inv_df_bits = get_compile_time_arg_val(15);  // 1 / 512
// (16: the compute's kLeader)
constexpr uint32_t kFullWorkers = get_compile_time_arg_val(17);  // M1L: full-layer workers (in activation order wf)
// M1L narrow / wide full layers: below kWideFrom DRAM tiles a full layer runs as step 2 did (the first kNarrowWorkers
// in wf order, K and V both on the reader, which runs ahead across layers); from it on, all kFullWorkers with K on the
// writer (NOC 0) and V on the reader (NOC 1) for the bandwidth. Sliding layers are always narrow (window-bounded).
constexpr uint32_t kWideFrom = get_compile_time_arg_val(18);
constexpr uint32_t kNarrowWorkers = get_compile_time_arg_val(19);
static_assert(R == 1 || R == 2 || R == 4 || R == 8, "attention takes 1 .. 8 rows");
static_assert(M >= 1 && M <= R, "rows");

constexpr uint32_t kGroups = 4, kGroupCores = 8;
constexpr uint32_t kSlidingWorkers = 7;     // G_g + 6 workers per sliding group
constexpr uint32_t kPage = 128, kRing = 1152, kWindow = 1024;
constexpr uint32_t kRingTiles = kRing / 32;  // 36
constexpr uint32_t kDts = 8, kDtf = 16;
constexpr uint32_t kBf8 = 1088, kBf16 = 2048, kRowTile = 64 * R;  // bytes: bf8 / bf16 32x32 tile, R x 32 tile
constexpr uint32_t kKW = 16;                // cb_k window per position tile (sliding uses the first 8)
constexpr uint32_t kPreK = get_compile_time_arg_val(20);  // M1L: K position tiles a wide worker's writer reads before
                                                          // its Q arrives (= cb_k's windows: they never wait on compute)
constexpr uint16_t kNegBig = 0xce6e;        // bf16 -1e9: masked scores (finite, so fully masked rows stay finite)

// semaphore ids on the attention cores (A_ERR / A_ERR2: this launch's table errors, read by the hub at the end)
constexpr uint32_t A_RX = 0, A_Q = 1, A_PARTS = 2, A_PADDR_L = 3, A_PADDR_G = 4, A_OADDR = 5, A_ERR = 6, A_ERR2 = 7;
// M1L: the reader's cb_ptab is loaded (the writer streams the workers' K tiles over NOC 0 from the same tables)
constexpr uint32_t A_PTAB = 8;

// status flags (M1_INTERFACE.md s5)
constexpr uint32_t F_POSITION = 0x1, F_SLOT = 0x2, F_TABLE_COLUMN = 0x4, F_PAGE = 0x8;
constexpr uint32_t kGroupsMax = 6;
// cb_ptab words: positions page [0, 32), slots page [32, 64), the slot's scratch pages [64, 72), then the slot's
// table rows, group g at kTabBase + g * P
constexpr uint32_t kPosW = 0, kSlotW = 32, kScratchW = 64, kTabBase = 80;

// CBs of the attention cores
constexpr uint32_t cb_q = 0;       // Q (bf16 32x32, CQ pages; L: packed by compute, others: multicast by L)
constexpr uint32_t cb_k = 1;       // K tiles per position tile (bf8, windows of kKW)
constexpr uint32_t cb_v = 2;       // V tiles per d (bf8, windows of NV)
constexpr uint32_t cb_wmask = 3;   // window masks of tiles lo, lo + 1 (bf16, persistent)
constexpr uint32_t cb_one = 4;     // [0] ones (reduce scaler), [1] identity (bf16, persistent)
constexpr uint32_t cb_s = 5;       // scores
constexpr uint32_t cb_p = 6;       // probabilities
constexpr uint32_t cb_m = 7;       // row max per Q tile
constexpr uint32_t cb_l = 8;       // row sum
constexpr uint32_t cb_own = 9;     // this core's partial (PT layout)
constexpr uint32_t cb_parts = 10;  // received partials (leaders, mergers)
constexpr uint32_t cb_pout = 11;   // merged partial (mergers)
constexpr uint32_t cb_out = 12;    // final O (roots)
constexpr uint32_t cb_sc = 13;     // merge scale factors
constexpr uint32_t cb_sc2 = 14;
constexpr uint32_t cb_diag = 15;
constexpr uint32_t cb_pos = 16;    // [0] p for the compute
// leaders
constexpr uint32_t cb_q32 = 17;    // raw Q rows (bf16 32x32, QTf x 16 pages)
constexpr uint32_t cb_kv32 = 18;   // raw K | V rows (sliding K 8, V 8; full KV 16)
constexpr uint32_t cb_w = 19;      // q norm weight tiles [0, Dt), k norm [16, 16 + Dt)
constexpr uint32_t cb_cs = 20;     // cos_s [0, 8), sin_s [8, 16), cos_f [16, 32), sin_f [32, 48) (persistent)
constexpr uint32_t cb_sq = 21;     // x^2 (fp32)
constexpr uint32_t cb_rinv = 22;   // 1 / rms (fp32)
constexpr uint32_t cb_xn = 23;     // normed rows (bf16)
constexpr uint32_t cb_kr = 24;     // roped K (bf16)
constexpr uint32_t cb_vn = 25;     // normed V (bf16)
constexpr uint32_t cb_k8 = 26;     // new K rows (bf8)
constexpr uint32_t cb_v8 = 27;     // new V rows (bf8)
constexpr uint32_t cb_tk = 28;     // tail K tiles c * Dt + d (bf8, 32 pages)
constexpr uint32_t cb_tv = 29;     // tail V tiles
constexpr uint32_t cb_tmask = 30;  // tail masks (bf16, persistent)
constexpr uint32_t cb_qready = 31; // Q packed (one page)
constexpr uint32_t cb_rx = 32;     // q|k|v rows from the streamers (raw R x 32 tiles)
constexpr uint32_t cb_ostage = 33; // O rows as R x 32 tiles for the streamers (raw)
constexpr uint32_t cb_ptab = 34;   // feed words and the slot's page tables (layout: kPosW .. kTabBase)
// M1L, full-layer workers: chunk partial, running partials (ping-pong); kOwn pages each
constexpr uint32_t cb_cp = 35, cb_ra = 36, cb_rb = 37;
constexpr uint32_t cb_k2 = 38;     // K tiles the reader streams (narrow layers; cb_k is the writer's: one producer each)
constexpr uint32_t kQ32 = QTf * kDtf, kCS = 48, kTail = 32;
constexpr uint32_t kOwn = PTs > PTf ? PTs : PTf;  // cb_own pages (every layer pushes all)

FORCE_INLINE bool is_full(uint32_t l) { return l % 6 == 5; }

// The step's schedule, the same on every core (from p)
struct Plan {
    uint32_t p;
    uint32_t tt0, ct;   // tail tiles [tt0, tt0 + ct)
    uint32_t s_lo, s_n, s_a;  // sliding: DRAM tiles [s_lo, s_lo + s_n) on s_a = min(7, s_n) workers
    uint32_t f_n, f_a;        // full: [0, f_n) on f_a workers (narrow: min(kNarrowWorkers, f_n); wide: kFullWorkers)
    bool wide;                // full layers in the wide mode (f_n >= kWideFrom)
};

FORCE_INLINE Plan make_plan(uint32_t p) {
    Plan pl;
    pl.p = p;
    pl.tt0 = p / 32;
    pl.ct = (p + M - 1) / 32 - pl.tt0 + 1;
    pl.s_lo = p >= kWindow - 1 ? (p - (kWindow - 1)) / 32 : 0;
    pl.s_n = pl.tt0 - pl.s_lo;
    pl.s_a = pl.s_n < kSlidingWorkers ? pl.s_n : kSlidingWorkers;
    pl.f_n = pl.tt0;
    pl.wide = pl.f_n >= kWideFrom;
    const uint32_t cap = pl.wide ? kFullWorkers : kNarrowWorkers;
    pl.f_a = pl.f_n < cap ? pl.f_n : cap;
    return pl;
}

// worker w of a active ones: tiles [first, first + count) of n
FORCE_INLINE void split(uint32_t n, uint32_t a, uint32_t w, uint32_t& first, uint32_t& count) {
    first = w * n / a;
    count = (w + 1) * n / a - first;
}

// tile t of the window needs the window mask (some row r < R sees positions after its start)
FORCE_INLINE bool needs_wmask(const Plan& pl, uint32_t t) { return pl.p + R - 1 > kWindow - 1 + 32 * t; }

// M1L full-layer tree (runtime args at `base`): 0 parent x | y << 16, 1 slot in the parent's parts, 2 the parent's
// parts mailbox (A_PADDR_L / A_PADDR_G), 3 merger (sends cb_pout), 4 children, 5 .. 11 their wf, 12 / 13 the skew
// before / through this worker (signed tiles), 14 the least f_n the skew applies at. Workers activate in wf order
// (wf < f_a), and every subtree's root has the smallest wf of its subtree, so a child is active iff its wf is:
// k = active children, the same count on the parent's writer and compute.
struct Tree {
    uint32_t base;
    uint32_t parent_xy() const { return get_arg_val<uint32_t>(base); }
    uint32_t slot() const { return get_arg_val<uint32_t>(base + 1); }
    uint32_t box() const { return get_arg_val<uint32_t>(base + 2); }
    bool merger() const { return get_arg_val<uint32_t>(base + 3) != 0; }
    uint32_t active_children(const Plan& pl) const {
        uint32_t k = 0;
        const uint32_t n = get_arg_val<uint32_t>(base + 4);
        for (uint32_t c = 0; c < n; c++) {
            k += get_arg_val<uint32_t>(base + 5 + c) < pl.f_a;
        }
        return k;
    }
    // this worker's position tiles of a full layer: the even split, moved by the cumulative skew once all
    // kFullWorkers are active and f_n leaves every range non-empty (cores whose partial passes more merges read less)
    void split_full(const Plan& pl, uint32_t wf, uint32_t& first, uint32_t& count) const {
        split(pl.f_n, pl.f_a, wf, first, count);
        if (pl.f_a == kFullWorkers && pl.f_n >= get_arg_val<uint32_t>(base + 14)) {
            const int32_t lo = static_cast<int32_t>(get_arg_val<uint32_t>(base + 12));
            const int32_t hi = static_cast<int32_t>(get_arg_val<uint32_t>(base + 13));
            first = static_cast<uint32_t>(static_cast<int32_t>(first) + lo);
            count = static_cast<uint32_t>(static_cast<int32_t>(count) + hi - lo);
        }
    }
};
constexpr uint32_t kTreeArgs = 15;

// step-2 full-layer worker order: group g holds [7g, 7g + 7), G_g first
FORCE_INLINE uint32_t full_group_active(const Plan& pl, uint32_t g) {
    const uint32_t lo = 7 * g;
    return pl.f_a > lo ? (pl.f_a - lo < 7 ? pl.f_a - lo : 7) : 0;
}
FORCE_INLINE uint32_t full_groups_active(const Plan& pl) {
    uint32_t n = 0;
    for (uint32_t g = 0; g < kGroups; g++) {
        n += full_group_active(pl, g) > 0;
    }
    return n;
}

// The launch's block (row 0 defines it: the step-2 attention serves one slot, rows 0 .. M - 1 at p .. p + M - 1)
struct Launch {
    uint32_t p, slot, flags;
    bool inert;
};
FORCE_INLINE Launch launch_of(
    volatile tt_l1_ptr uint32_t* pos, volatile tt_l1_ptr uint32_t* slot, uint32_t capacity, uint32_t max_users) {
    Launch L{0, 0, 0, true};
    const int32_t p0 = static_cast<int32_t>(pos[0]);
    const uint32_t s = slot[0];
    if (p0 < 0) {
        return L;  // inert: nothing to decode, nothing flagged
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
    L.inert = false;
    return L;
}

#ifndef COMPILE_FOR_TRISC
// layer l's group (4 bits per layer, 8 layers per word, from `gmap` runtime args)
struct Tables {
    volatile tt_l1_ptr uint32_t* w;  // cb_ptab
    uint32_t P;                      // table columns (capacity / 128)
    uint32_t gmap_arg, pool_arg;     // runtime arg indices: 8 packed group words, kGroupsMax pool page counts
    bool inert;
    uint32_t group(uint32_t l) const { return (get_arg_val<uint32_t>(gmap_arg + l / 8) >> (4 * (l % 8))) & 0xf; }
    uint32_t scratch(uint32_t g) const { return w[kScratchW + g]; }
    // physical page of position tile t in layer l (bounds checked; errors -> the scratch page, flagged)
    uint32_t page(uint32_t l, uint32_t t) const {
        const uint32_t g = group(l);
        if (inert) {
            return scratch(g);
        }
        const uint32_t col = t / 4;
        if (col >= P) {
            record(F_TABLE_COLUMN, col, 0);
            return scratch(g);
        }
        const uint32_t e = w[kTabBase + g * P + col];
        if (e >= get_arg_val<uint32_t>(pool_arg + g)) {
            record(F_PAGE, col, e);
            return scratch(g);
        }
        return e;
    }
    // tile id of (position tile t, head, d) in a [pages, H, 128, D] BF8 cache
    uint32_t kv_tile(uint32_t l, bool full, uint32_t t, uint32_t head, uint32_t d) const {
        const uint32_t H = full ? 1 : 4, Dt = full ? kDtf : kDts;
        return ((page(l, t) * H + head) * 4 + t % 4) * Dt + d;
    }
    static void record(uint32_t flag, uint32_t col, uint32_t entry) {
        volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_ERR));
        volatile tt_l1_ptr uint32_t* e2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(A_ERR2));
        if ((*e & 0xf) == 0) {
            *e = flag | ((col < 4095 ? col : 4095) << 9);  // row 0 (bits 4..8): the block's first row
            *e2 = entry;
        } else {
            *e = *e | flag;
        }
    }
};
#endif

}  // namespace gatt
