// MKA2: the attention-type layer inside the megakernel engine (MK1 engine + the MKA1 SDPA phase), spec v0.3.
// Per layer L, every core of the full grid (STREAM rect time-multiplexed, plus the ATTN-only cores):
//   [STREAM] MIXER (qkv in-proj stand-in, real bytes)  ->  SDPA (phase type 5, ATTN CBs 9-12, all cores)
//   -> [STREAM] wo stand-in + AR stall -> MLP gate|up / down (fm11 engine) -> AR stall.
// ATTN synchronisation uses GLOBAL semaphores (spec v0.1): uniform L1 addresses passed in runtime args; all counters
// are cumulative across layers (value for layer L = (L + 1) x per-layer count), so nothing is reset inside a pass.
#pragma once

// MKA_EXT_LEANZ (profiled runs only): the EXT (owner) data-movement kernels keep only the zones the layer / AR reports
// use (W-OREL, W-WOAR, the MK_AR_ZONE ones); the rest compile out. The profiler zones alone put the owners-only EXT
// binary over the 70656 B kernel config buffer under GDN24 + RWO + MK_REAL_AR + MK_REAL_AAR (fits at O2 without them).
// MKA_STREAM_LEANZ: the same on the STREAM DM kernels (MKZK zones stay: W-ARWAIT = the layer report's boundary)
#define MKZK(n) DeviceZoneScopedN(n)
#if (defined(MKA_ROLE_EXT) && defined(MKA_EXT_LEANZ)) || (defined(MKA_ROLE_STREAM) && defined(MKA_STREAM_LEANZ))
#define MKZ(n)
#else
#define MKZ(n) DeviceZoneScopedN(n)
#endif
#include "../../kernels/fm11_common.hpp"
constexpr uint32_t CB_WM = 6, CB_YM = 7, CB_GO = 8;
constexpr uint32_t CB_Q = 9, CB_K = 10, CB_V = 11, CB_O = 12;
constexpr uint32_t CB_ONE = 24, CB_P = 25, CB_R = 26, CB_D = 27, CB_OA = 28, CB_SCR = 29;
// MKA_RQKV (real qkv PROJ, spec v0.4 deferred norm + v0.5 STREAM_EXT): rstd tile, fp32 partial/acc, bf16 output,
// partner partial receive slot, STREAM_EXT weight ring, AR owners' Σx² partials (CB 20 = CB_RSTD)
constexpr uint32_t CB_RS = 13, CB_QA = 14, CB_QO = 17, CB_QR = 18, CB_PQ = 19, CB_RSP = 20;
// MKA_TREEFLAG: per-child arrival flags (2 x FANOUT words) at a uniform L1 address, so a parent pulls each child as
// soon as THAT child is done instead of after the slowest one
constexpr uint32_t CB_TF = 31;
// MKA_RWO (real wo, E'): A rows (48 bf16 tiles), sig(gate) (48 tiles, aliased in place by A (.) S), owner CB_XO (10 tiles)
#ifndef MKA_WO_MAXT
#define MKA_WO_MAXT 5   // wo output tiles per wo core (36 ext cores: 5; GDN24 bring-up, 16 owners: 10)
#endif
constexpr uint32_t CB_A = 15, CB_S = 21, CB_AS = 22, CB_XO = 23, WK = 48, WN = 160, WO_MAXT = MKA_WO_MAXT;
// ext ring (CB 19) pages: the 2-tile full-K qkv slice (pre-QSK) or the Wo slice (WO_MAXT x WK), whichever is larger
constexpr uint32_t RING_P = (WO_MAXT * WK > 320) ? WO_MAXT * WK : 320;
// MKA_WORING: the Wo slice gets its own ring (CB 31, a full 5 x 48-page slot) filled during the GDN-type layers before
// its attention layer (spec §18 frees the qkv ring; this emulates it), so no Wo byte lands in the SDPA window
constexpr uint32_t CB_WOR = 31, WOR_SLOT = WO_MAXT * WK;
// MKA_RQNR head cores: CB_A slots 0-7 = this head's 8 qkv tiles (rstd-scaled); CB 31 = fp32 scratch (x^2, then y);
// CB 30 = constants (gamma tiles 0-7 with every row = gamma, cos, sin); output 8 tiles in CB_XO (full 10-page slot)
constexpr uint32_t CB_HSQ = 31, CB_HG = 30, NHEAD = 7;
constexpr uint32_t NRF = 3;
// KV cache addressing. MKA_KVPROD (prod vLLM): per layer TWO tensors paged_k / paged_v [num_blocks, n_kv_heads,
// block_size, 256] TILE bf8, tile page of (block p, head h, pos-tile r, dim-tile c) = ((p * NKVH + h) * BS32 + r) * 8 + c.
// SDPA block b (32 positions) of a user maps to (p = user * BPU + b / BS32, r = b % BS32); the bench's page table is
// the identity (user-major blocks). Without MKA_KVPROD: one tensor per layer, K / V interleaved per 32-position block.
#if defined(MKA_KVPROD)
constexpr uint32_t KV_NKVH = MKA_NKVH, KV_BS32 = MKA_BS32, KV_H = MKA_KVH;
FORCE_INLINE uint32_t kv_page(uint32_t user, uint32_t blk, uint32_t bpu, bool /*v*/) {
    const uint32_t p = user * bpu + blk / KV_BS32, r = blk % KV_BS32;
    return ((p * KV_NKVH + KV_H) * KV_BS32 + r) * 8;
}
#else
FORCE_INLINE uint32_t kv_page(uint32_t user, uint32_t blk, uint32_t nbk, bool v) { return ((user * nbk + blk) * 2 + (v ? 1 : 0)) * 8; }
#endif   // full-grid multicast rectangles (left block, right block rows 0-7, right block rows 8-9 minus AR corner)   // ATTN scratch range 24-31 (spec)
// MKA_KVA (stream E, spec v0.9.16): paged KV append of all users' T rows into the prod paged_k / paged_v (BFP8) on the
// 16 owners: owner (10, c) appends K dim tile c, (11, c) V dim tile c. The source bf16 tile (k head CB_XO tile c / qkv
// tile 56 + c) is written into CB_KVS by its producer; compute packs it to BFP8 (CB_KVB page 0: the packer conversion of
// paged_update_cache). BFP8 rows are independent (one exponent per 16-datum face row), so the writer splices each row
// (2 exponent bytes + 2 x 16 mantissa bytes) into the target tiles prefetched into CB_KVB pages 1.. and writes them back.
// Defines (engine/program.py): MKA_KVA_NU users, MKA_KVA_T rows per user, MKA_KVA_NB page-table width, MKA_KVA_POS /
// MKA_KVA_PT kvpos (int32 [NU*T]) / page table (int32 [NU, NB]) DRAM addresses, MKA_KVA_PTS page-table stick bytes,
// MKA_KVA_GS {done, src, addr} global semaphores, MKA_KVA_KXY / VXY append-core NoC xy (x | y << 8), MKA_KVA_WXY /
// MKA_KVA_NW waiters of KVA_DONE (the Q root first, then the readers of each own user's last 2 KV blocks).
#if defined(MKA_KVA)
#if !defined(MKA_KVPROD) || !defined(MKA_RQNR)
#error "MKA_KVA needs MKA_KVPROD (prod paged caches) and MKA_RQNR (the k head output)"
#endif
constexpr uint32_t CB_KVS = 6, CB_KVB = 8;   // owners only (both ids are STREAM-rect-only otherwise)
constexpr uint32_t KVA_NU = MKA_KVA_NU, KVA_T = MKA_KVA_T, KVA_R = KVA_NU * KVA_T, KVA_NP = 1 + MKA_KVA_PPU * KVA_NU;   // PPU: target tiles per user (verify: 2)
constexpr uint32_t KVA_GS_DONE = MKA_KVA_GS_DONE, KVA_GS_SRC = MKA_KVA_GS_SRC, KVA_GS_AD = MKA_KVA_GS_AD;
constexpr uint32_t KVA_PER_LAYER = 16;   // append cores incrementing each waiter per attention layer
constexpr uint32_t KVA_TRID = 15;
// KVA_DONE waiters (the readers of each own user's last 2 KV blocks) are always non-STREAM cores (the RMASK member order
// puts each user's last block on an ext core; engine/program.py asserts it), so the wait code compiles out of STREAM
#if !defined(MKA_ROLE_STREAM)
#define MKA_KVA_WAITER 1
#endif
#if defined(MKA_KVA_PROF)   // KV append zones (W-KVAT / W-KVAS / W-KVA / R-KVAW) are opt-in: code budget (STREAM / EXT text)
#define KVA_ZONE(n) DeviceZoneScopedN(n)
#else
#define KVA_ZONE(n)
#endif
static_assert(KVA_R <= 32, "KV append: users x T rows must fit one tile");
#endif
// qkv roles: 0 none, 1 K-half owner (STREAM), 2 K-half partner (STREAM), 3 STREAM_EXT (2 tiles, full K)
constexpr uint32_t QR_NONE = 0, QR_OWNER = 1, QR_PARTNER = 2, QR_EXT = 3;
// CB_QO slot: qkv output (1-2 tiles) and, under MKA_RWO, the wo output (4-5 tiles) share CB 17; every use reserves /
// pushes / pops the FULL slot so the ring never straddles its end (same rule as the KV groups)
#if defined(MKA_RWO)
constexpr uint32_t QO_SLOT = WO_MAXT;
#else
constexpr uint32_t QO_SLOT = 0;   // 0: use the exact tile count
#endif
// STREAM_EXT ring split (MKA_PQW): the writer (NOC0) streams pages [0, PQ_WS) of the next layer's slice, the reader
// (NOC1) [PQ_WS, 2 KT). 0 = all reader, 1 = all writer, 2 = half / half
#ifndef MKA_PQW
#define MKA_PQW 0
#endif
constexpr uint32_t PQ_QSL = 320, PQ_WS = MKA_PQW == 0 ? 0 : (MKA_PQW == 1 ? PQ_QSL : PQ_QSL / 2);
#ifndef MKA_NU
#define MKA_NU 4
#endif
#ifndef MKA_FAN
#define MKA_FAN 6
#endif
constexpr uint32_t DH = 8, NO = DH + 1, QT = 8 * MKA_NU, FANOUT = MKA_FAN, NUSR = MKA_NU;   // own users per chip (u8: 4, u1: 1)
// global-semaphore slots (index into the runtime-arg address array)
constexpr uint32_t GS_QFLAG = 0, GS_LVL0 = 1, GS_LVL1 = 2, GS_DONE = 3, GS_MIXDONE = 4, GS_ADONE = 5, GS_ADFLAG = 6, GS_KVPROG = 7, GS_QPART = 8, GS_XRDY = 9, GS_PQH = 10, GS_ACNT = 11, GS_AFLAG = 12, GS_WOCNT = 13, GS_WODONE = 14, GS_AADDR = 15,
                   GS_SADDR = 16, GS_SCNT = 17, GS_SFLAG = 18, GS_WOHC = 19, GS_RADDR = 20, GS_HCNT = 21, GS_QHD = 22, NGS = 23;   // KVPROG: core-local count of KV groups the reader pushed (kv_mode 2)
FORCE_INLINE volatile tt_l1_ptr uint32_t* l1w(uint32_t addr) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr); }
#if defined(MKA_RQPOS)
// MKA_RQPOS (real per-user positions, spec v1.8x): the engine reads this replay's in-trace tensors through G's 'replay'
// arg block (engine/argblock.py, L1 at MKA_REPLAY_L1, payload from word 4: kvpos, ctrl, cos, sin, pt_rows, pt_users,
// pt_groups) and the model-global 'attn' block (MKA_ATTN_L1: pt_addr, pt_B, pt_nb, n_attn, (k, v) per attention layer,
// block_size, n_kv_heads). kvpos: int32 [32] RM DRAM (row u*T + j, -1 = HOLD / idle); cos / sin: [1,1,32,64] bf16 TILE
// DRAM (tile 0 = the 32 frequencies of each row, prod's values). Page table: _vfy_kvpt_users int32 [B, nb] RM DRAM.
// Per-core table at MKA_KVP_TAB (uniform L1 address): [0] info = bit 31 valid | nv (bits 0-10: valid KV blocks of this
// core) | m0 (bits 11-20: local index of the user's last valid block, 0x3FF none) | m1 (bits 21-30: the block before it
// when the user's T rows straddle); [1] b0 (first block; MKA_RQDYN: the static share rescaled to the user's length);
// [2] pt window base (entry index of word KVP_PT); [3 .. 3+T) per-row causal limits (HOLD rows: the user's max);
// [KVP_PT ..] this core's page-table window. The reader writes it at pass start (info last) and resets [0] at program
// end; writer / compute wait [0]. Mask tiles (bf16 0 / -9984, ADDITIVE on S before exp, so stale positions can't
// overflow exp) are built once per pass by the WRITER into CB_ONE pages 1 (m0) / 2 (m1) (CB_ONE = 3 pages under
// RQPOS; page 0 = ones). CB_OA must not carry them: on a tree root it flows compute -> reader (normalize), and a CB used
// in both directions deadlocks (ragged users put the last block on the root).
// Own users (MKA_UPAR): global user g = 2u + MKA_UPAR_PAR (prod kv_split ownership).
constexpr uint32_t KVP_NONE = 0x3FF, KVP_PT = MKA_KVP_PT;   // host: MKA_KVP_TAB + 4 * KVP_PT is 64 B aligned (DRAM reads)
FORCE_INLINE uint32_t rq_replay(uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKA_REPLAY_L1)[4 + i]; }
FORCE_INLINE uint32_t rq_attn(uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MKA_ATTN_L1)[4 + i]; }
#if defined(MKA_UPAR)
FORCE_INLINE uint32_t rq_g(uint32_t u) { return 2 * u + MKA_UPAR_PAR; }
#else
FORCE_INLINE uint32_t rq_g(uint32_t u) { return u; }
#endif
FORCE_INLINE uint32_t kvp_wait() {   // DM side: the reader's word for this pass
    volatile tt_l1_ptr uint32_t* w = l1w(MKA_KVP_TAB);
    uint32_t v;
    do { invalidate_l1_cache(); v = *w; } while (!(v >> 31));
    return v;
}
#if defined(MKA_KVPROD)
// paged K / V tile page of SDPA block blk (32 positions) through this core's page-table window
FORCE_INLINE uint32_t kv_page_pt(uint32_t blk) {
    volatile tt_l1_ptr uint32_t* t = l1w(MKA_KVP_TAB);
    const uint32_t p = t[KVP_PT + blk / KV_BS32 - t[2]], r = blk % KV_BS32;
    return ((p * KV_NKVH + KV_H) * KV_BS32 + r) * 8;
}
#endif
#endif
FORCE_INLINE void spin_cycles(uint32_t cyc) {
    if (cyc == 0) return;
    const uint32_t t0 = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    while ((uint32_t)(reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L) - t0) < cyc) {}
}
// multicast an L1 word holding `val` to the same address on up to 2 rects (loopback on self_rect)
// NR: number of rectangles in rc (full grid: 3, since spec v0.9.1 cuts the AR sender corner out of the right block)
template <uint32_t NR = 2>
FORCE_INLINE void mcast_word(uint32_t addr, uint32_t val, const uint32_t (*rc)[5], uint32_t self_rect, uint8_t noc) {
    *l1w(addr) = val;
    for (uint32_t i = 0; i < NR; ++i) {
        if (rc[i][4] == 0) continue;
        const uint64_t m = noc == 0 ? get_noc_multicast_addr(rc[i][0], rc[i][1], rc[i][2], rc[i][3], addr, noc)
                                    : get_noc_multicast_addr(rc[i][2], rc[i][3], rc[i][0], rc[i][1], addr, noc);
        if (i == self_rect) noc_semaphore_set_multicast_loopback_src(addr, m, rc[i][4], false, noc);
        else noc_semaphore_set_multicast(addr, m, rc[i][4], false, noc);
    }
}

// Mixed layer groups (MK mixed): bit L of MKA_AMASK = 1 -> attention-type layer, 0 -> GDN-type layer (GDN step +
// out-proj modelled as a MKA_S_GDN-cycle stall on the STREAM chain until the -mk-gdn phases land). Attention
// counters (Q flag, MIXDONE, ADONE, ADFLAG, tree sems) are cumulative over ATTENTION layers only: aidx(L).
#ifndef MKA_AMASK
#define MKA_AMASK 0xFFFFFFFFu
#endif
#ifndef MKA_S_GDN
#define MKA_S_GDN 0
#endif
FORCE_INLINE bool is_attn(uint32_t L) { return ((uint32_t)(MKA_AMASK) >> L) & 1u; }
FORCE_INLINE uint32_t aidx(uint32_t L) { return (uint32_t)__builtin_popcount((uint32_t)(MKA_AMASK) & ((1u << L) - 1u)); }
// MKA_W_ORD (integ repack order): the qkv / wo weight tensors hold the ATTENTION layers only, attention ordinal a at page
// a * stride. wla(L) = the page-block index of engine layer L: aidx(L) for an attention layer; a GDN layer's stand-in reads
// (QSK slot M in-proj stand-in, bytes only) clamp to the last attention layer.
FORCE_INLINE uint32_t wla(uint32_t L) {
#if defined(MKA_W_ORD)
    constexpr uint32_t NA = (uint32_t)__builtin_popcount((uint32_t)(MKA_AMASK));
    const uint32_t a = aidx(L);
    return a < NA ? a : NA - 1;
#else
    return L;
#endif
}
