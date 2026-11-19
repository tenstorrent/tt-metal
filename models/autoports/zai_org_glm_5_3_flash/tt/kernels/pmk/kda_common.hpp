// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Prefill megakernel (Stage 16, doc/ttnn/16-prefill-production/prefill_megakernel.md) — the KDA phase (P4 conv + P5
// chunk prepare on the ITEM cores, P6 chunk scan + gated RMSNorm on the SCAN cores, P9 state write). Host side:
// tt/pmk_kda.py (documents the design). CB indices follow the contract's §4b map: KDA prepare from PK_ITEM_CB0 (32),
// KDA scan from PK_SCAN_CB0 (50); the item and scan core sets are disjoint, so the item set may run past 49 (to 54).
//
// Hand-offs (§4b): H3 q/k/v/g/β (SEM_QKV = 5, raised per 32-row block by the P3 / P4 stream phases) -> item cores;
// the items' chunk terms -> DRAM scratch TERMS, ONE counter SEM_KDA (6) on the head's scan cores, made exact by the
// item writers signalling in chunk order (a baton: the writer of chunk c waits for the writer of chunk c - 1, the same
// id 6 on the item cores, before it raises the scan cores' counter; dead chunks form a suffix and end the chain), so a
// scan core waits for chunk c with SEM_KDA >= c + 1; the gated-norm partial sums between a head's SPREAD scan cores
// (SEM_RMS, kda_rec's exchange); H4 `on` (SEM_ON = 7) raised on the P7 consumers by the leader scan core after every
// chunk all scan cores finished (SEM_ONAGG counts them). Semaphore ids are 0..15 (NUM_SEMAPHORES): SEM_RMS / SEM_ONAGG
// default to 0 / 1, free on MM-only cores (the mHC pre ids of members / roots): the scan cores must be MM-only cores,
// or the family passes PK_SEM_RMS / PK_SEM_ONAGG. Every wait is a wait_min on a monotonic counter; the family resets
// them once per layer.
#pragma once

#include <cstdint>

#ifndef PK_ITEM_CB0
#define PK_ITEM_CB0 32
#endif
#ifndef PK_SCAN_CB0
#define PK_SCAN_CB0 50
#endif

namespace pk {
// the integrator's per-CB index map (Stage 16, pmk_sparse_kda.kda_bodies): PK_ITEM_MAP / PK_SCAN_MAP = comma lists of
// the ITEM_CBS / SCAN_CBS indices (free indices of the member / MM-only composites, which also host other pieces' CBs);
// default: contiguous from PK_ITEM_CB0 / PK_SCAN_CB0 (the standalone unit)
#ifdef PK_ITEM_MAP
constexpr uint32_t ITEM_IDX[] = {PK_ITEM_MAP};
#define PK_ICB(k) (ITEM_IDX[k])
#else
#define PK_ICB(k) (PK_ITEM_CB0 + (k))
#endif
#ifdef PK_SCAN_MAP
constexpr uint32_t SCAN_IDX[] = {PK_SCAN_MAP};
#define PK_SCB(k) (SCAN_IDX[k])
#else
#define PK_SCB(k) (PK_SCAN_CB0 + (k))
#endif

constexpr uint32_t T32 = 4096, T16 = 2048;
constexpr uint32_t CHUNK = 32;

// semaphores (the coordinator's map for non-lane cores: 0..3 mHC pre, 4 HN, 5 QKV, 6 KDA, 7 ON, 8 A, 9 M, 10 / 11
// PARK, 12 LOGITS, 13 TAIL, 14 / 15 FabricStream; ids are 0..15)
#ifndef PK_SEM_RMS
#define PK_SEM_RMS 0
#endif
#ifndef PK_SEM_ONAGG
#define PK_SEM_ONAGG 1
#endif
constexpr uint32_t SEM_QKV = 5;           // H3: P3 / P4 blocks landed (item cores)
constexpr uint32_t SEM_KDA = 6;           // scan cores: chunks signalled (in order); item cores: the signalling baton
constexpr uint32_t SEM_ON = 7;            // H4: `on` chunks done (the P7 consumers)
constexpr uint32_t SEM_RMS = PK_SEM_RMS;  // scan cores: peers' gated-norm partial sums landed
constexpr uint32_t SEM_ONAGG = PK_SEM_ONAGG;  // scan cores -> the leader scan core, chunk done (aggregated into SEM_ON)

// ---------------------------------------------------------------- item cores (P4 conv + P5 prepare)
constexpr uint32_t CB_U = PK_ICB(0);      // bf16 u (q|k|v pre-conv) tiles, 24 per item: [prev chunk 12 | this chunk 12]
constexpr uint32_t CB_F = PK_ICB(1);      // fp32 raw f_b output tiles (Kt) of the item
constexpr uint32_t CB_BETA = PK_ICB(2);   // fp32 column tile: col 0 = b (pre-sigmoid) rows; + control words (CTL_COL)
constexpr uint32_t CB_MASK = PK_ICB(3);   // fp32 column tile: 1.0 for rows < the chunk's valid rows
constexpr uint32_t CB_CONST = PK_ICB(4);  // fp32 constants (K_*)
constexpr uint32_t CB_TAPS =
    PK_ICB(5);                          // bf16 row-0 tiles: tap j of column tile t at page j * 3 Kt + t (head resident)
constexpr uint32_t CB_DTR = PK_ICB(6);  // fp32 row-0 tiles: dt_bias 0..Kt-1, decay rate Kt..2Kt-1 (head resident)
constexpr uint32_t CB_CONV = PK_ICB(7);  // bf16 silu(conv): q 0..Kt-1 | k | v
constexpr uint32_t CB_G = PK_ICB(8);     // fp32 gate g = lb * sigmoid((f + dt) * rate), masked
constexpr uint32_t CB_NQ = PK_ICB(9);    // fp32 normalized q (scaled)
constexpr uint32_t CB_NK = PK_ICB(10);   // fp32 normalized k
constexpr uint32_t CB_W0 = PK_ICB(11);   // fp32 workspaces (prepare_chunk_recurrence's workspace_0..3)
constexpr uint32_t CB_W1 = PK_ICB(12);
constexpr uint32_t CB_W2 = PK_ICB(13);
constexpr uint32_t CB_W3 = PK_ICB(14);
constexpr uint32_t CB_TW0 = PK_ICB(15);  // fp32 1-tile workspaces (tile_workspace_0..2; TW1 has 2 entries)
constexpr uint32_t CB_TW1 = PK_ICB(16);
constexpr uint32_t CB_TW2 = PK_ICB(17);
constexpr uint32_t CB_SD = PK_ICB(18);    // fp32 scan_decay = exp(G)
constexpr uint32_t CB_CID = PK_ICB(19);   // fp32 centered inverse decay
constexpr uint32_t CB_ANCH = PK_ICB(20);  // fp32 anchor decay exp(G_last / 2)
constexpr uint32_t CB_AKK = PK_ICB(21);   // fp32 Akk (1 tile)
constexpr uint32_t CB_OUT = PK_ICB(22);   // fp32 chunk terms, OUT_PAGES per item (2 items deep), order below
#ifdef PMK_ITEM_CONV_FPU
// bf16 S1..S3 (the reader's row-shifted u tiles, page (s - 1) CT + t), ONE item deep: split off CB_U's two-deep ring
// (5 CT -> 2 CT pages per item there; -72 KB per member at Kt 4: the fused B7g census, coordinator 03:23:34Z)
constexpr uint32_t CB_SH = PK_ICB(23);
constexpr uint32_t ITEM_CBS = 24;
#else
constexpr uint32_t ITEM_CBS = 23;
#endif

// constants (CB_CONST pages)
constexpr uint32_t K_EYE = 0, K_TRIL = 1, K_ONES = 2, K_MASK0 = 3, K_SDN1 = 6, K_SUP1 = 9, K_PAGES = 12;
// control words: column tiles carry them in face 1 (row 0, col 16: never read by a column broadcast), row tiles in
// face 2 (row 16, col 0: never read by a row broadcast); uint32 element offsets
constexpr uint32_t CTL_COL = 256, CTL_ROW = 512;

// chunk-term page offsets within an item of CB_OUT and of the TERMS scratch (Kt key tiles, VTF value tiles of a head)
constexpr uint32_t off_vbeta(uint32_t, uint32_t) { return 0; }
constexpr uint32_t off_qdec(uint32_t, uint32_t vtf) { return vtf; }
constexpr uint32_t off_kd(uint32_t kt, uint32_t vtf) { return vtf + kt; }
constexpr uint32_t off_intra(uint32_t kt, uint32_t vtf) { return vtf + 2 * kt; }
constexpr uint32_t off_tinv(uint32_t kt, uint32_t vtf) { return vtf + 2 * kt + 1; }
constexpr uint32_t off_fd(uint32_t kt, uint32_t vtf) { return vtf + 2 * kt + 2; }
constexpr uint32_t off_kdt(uint32_t kt, uint32_t vtf) { return vtf + 3 * kt + 2; }
constexpr uint32_t out_pages(uint32_t kt, uint32_t vtf) { return vtf + 4 * kt + 2; }

// ---------------------------------------------------------------- scan cores (P6 + gated norm, P9)
constexpr uint32_t CB_S = PK_SCB(0);   // fp32 S (Kt x Vt tiles of this v-block), read once
constexpr uint32_t CB_IN = PK_SCB(1);  // fp32 chunk terms + gate_out per chunk (IN_PAGES), 2 chunks deep
constexpr uint32_t CB_W = PK_SCB(2);   // fp32: page 0 = o_norm row tile of the v-block (+ CTL_ROW words), page 1 = ones
constexpr uint32_t CB_SR = PK_SCB(3);  // fp32 state ring (2 x Kt Vt)
constexpr uint32_t CB_VN = PK_SCB(4);  // fp32 corrected value U (Vt)
constexpr uint32_t CB_OI = PK_SCB(5);  // fp32 intermediate (Vt)
constexpr uint32_t CB_SCR = PK_SCB(6);    // fp32 scratch (Kt Vt)
constexpr uint32_t CB_SU = PK_SCB(7);     // fp32 state update (Kt Vt)
constexpr uint32_t CB_ST = PK_SCB(8);     // fp32 decayed state (Kt Vt)
constexpr uint32_t CB_FS = PK_SCB(9);     // fp32 final state (Kt Vt) -> P9
constexpr uint32_t CB_O = PK_SCB(10);     // fp32 chunk output y (Vt), pre-norm
constexpr uint32_t CB_PSUM = PK_SCB(11);  // fp32 partial sum of squares (column tile), sent to the peers
constexpr uint32_t CB_PIN = PK_SCB(12);   // fp32 peers' partials, 2 x (SPREAD - 1) pages
constexpr uint32_t CB_ON = PK_SCB(13);    // bf16 gated-normed output tiles (2 x Vt) -> H4
constexpr uint32_t SCAN_CBS = 14;

// IN ring page offsets per chunk (this core's Vt value tiles; Kt key tiles)
constexpr uint32_t in_vbeta(uint32_t, uint32_t) { return 0; }
constexpr uint32_t in_kd(uint32_t, uint32_t vt) { return vt; }
constexpr uint32_t in_qdec(uint32_t kt, uint32_t vt) { return vt + kt; }
constexpr uint32_t in_intra(uint32_t kt, uint32_t vt) { return vt + 2 * kt; }
constexpr uint32_t in_tinv(uint32_t kt, uint32_t vt) { return vt + 2 * kt + 1; }
constexpr uint32_t in_fd(uint32_t kt, uint32_t vt) { return vt + 2 * kt + 2; }
constexpr uint32_t in_kdt(uint32_t kt, uint32_t vt) { return vt + 3 * kt + 2; }
constexpr uint32_t in_go(uint32_t kt, uint32_t vt) { return vt + 4 * kt + 2; }
constexpr uint32_t in_pages(uint32_t kt, uint32_t vt) { return 2 * vt + 4 * kt + 2; }

// element index of (row r, col c) within a 32 x 32 tile (16 x 16 faces f0 f1 / f2 f3)
constexpr uint32_t tile_off(uint32_t r, uint32_t c) {
    return (((r >> 4) << 1) + (c >> 4)) * 256u + (r & 15u) * 16u + (c & 15u);
}

// valid_end -> chunks with any valid row (the chronology of chronology.hpp at sp_size 1: chunk_count / partial_*)
constexpr uint32_t valid_chunks(uint32_t ve) { return (ve + CHUNK - 1) / CHUNK; }
constexpr uint32_t chunk_rows(uint32_t ve, uint32_t c) {  // valid rows of chunk c (0 .. 32)
    return ve >= (c + 1) * CHUNK ? CHUNK : (ve > c * CHUNK ? ve - c * CHUNK : 0);
}

// in-program probe rows (the unit reads them back): one 128-B row per core of a [n_item + n_scan, 32] uint32 ROW_MAJOR
// DRAM tensor; the data-movement RISCs write 16 words each (NCRISC words 0..15, BRISC words 16..31) at their end:
// [0] magic, [1] valid_end as read, [2] nvalid / items, [3] units done, [4..] checksums (sums of uint32 words)
constexpr uint32_t PROBE_ROW_BYTES = 128;
constexpr uint32_t PROBE_ITEM_R = 0xAB01, PROBE_ITEM_W = 0xAB02, PROBE_SCAN_R = 0xAB03, PROBE_SCAN_W = 0xAB04;
FORCE_INLINE uint32_t probe_sum(uint32_t l1, uint32_t words) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    uint32_t a = 0;
    for (uint32_t i = 0; i < words; ++i) {
        a += p[i];
    }
    return a;
}

}  // namespace pk
