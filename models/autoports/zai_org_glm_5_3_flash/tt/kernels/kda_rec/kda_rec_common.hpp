// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7a kda_rec (tt/kda_rec.py): the KDA decode recurrence (ttnn.experimental.kda.fused_recurrent_kda's kernels,
// decode T = 1) + the sigmoid-gated o RMSNorm, one head per core. CB indices: 14 at REC_CB0 + n and the three extra
// (reader scratch, norm weight, writer output assembly) at REC_CB_X0 + n, so a megakernel can place them in two free
// index ranges.
#pragma once

#include <cstdint>

#ifndef REC_CB0
#define REC_CB0 0
#endif
#ifndef REC_CB_X0
#define REC_CB_X0 (REC_CB0 + 14)
#endif

#ifndef REC_HEADS
#define REC_HEADS \
    64  // heads of the layer shard (Stage 14 TP8: 8): cores h >= REC_HEADS of the core list skip the phase
#endif

// REC_SPREAD (Stage 14 TP: a chip's few heads over more cores): cores per head; core c of the list serves head
// c / REC_SPREAD, value-column block c % REC_SPREAD (Vt / REC_SPREAD tiles of S, v, gate_out, o). The recurrence
// is column-independent; only the gated RMSNorm's sum of squares crosses cores: each core sends its partial (one
// value per item) to the head's other cores (REC_RMS_SEM counts them).
#ifndef REC_SPREAD
#define REC_SPREAD 1
#endif
#ifndef REC_CB_PSUM
#define REC_CB_PSUM 30  // fp32 partial sum of squares (1 page), packed by compute, sent to the peers by the writer
#endif
#ifndef REC_CB_PIN
#define REC_CB_PIN (REC_CB_X0 + 3)  // the peers' partials, 2 x (S - 1) pages (shares the debug tap's index)
#endif
#ifndef REC_RMS_SEM
#define REC_RMS_SEM 13  // program semaphore: the peers' partials landed (MkLayer uses 0..12)
#endif
#if REC_SPREAD > 1 && defined(REC_DEBUG)
#error "REC_SPREAD > 1 uses the debug tap's CB index"
#endif

#ifndef REC_DEPTH
#define REC_DEPTH 2  // per-item input / state CBs: 2 = double-buffered (standalone), 1 = lean (megakernel owners)
#endif

namespace kda_rec {
constexpr uint32_t CB_Q = REC_CB0 + 0;      // fp32 [K,1] column tiles of one item's q (reader)
constexpr uint32_t CB_K = REC_CB0 + 1;      // fp32 [K,1]
constexpr uint32_t CB_V = REC_CB0 + 2;      // fp32 [1,V] row tiles
constexpr uint32_t CB_G = REC_CB0 + 3;      // fp32 [K,1] log decay
constexpr uint32_t CB_BETA = REC_CB0 + 4;   // fp32 1 tile, beta replicated over row 0
constexpr uint32_t CB_S0 = REC_CB0 + 5;     // fp32 [K,V] state of one item (read)
constexpr uint32_t CB_QCOL = REC_CB0 + 6;   // fp32 [K,1] normalised, scaled q
constexpr uint32_t CB_KCOL = REC_CB0 + 7;   // fp32 [K,1] normalised k
constexpr uint32_t CB_KFULL = REC_CB0 + 8;  // fp32 [K,32] normalised k broadcast over columns
constexpr uint32_t CB_DCOL = REC_CB0 + 9;   // fp32 [K,1] exp(g)
constexpr uint32_t CB_ST = REC_CB0 + 10;    // fp32 [K,V] updated state (written back in place)
constexpr uint32_t CB_O = REC_CB0 + 11;     // fp32 [1,V] o row (compute scratch between the step and the norm)
constexpr uint32_t CB_GO = REC_CB0 + 12;    // fp32 [1,V] gate_out row of the item (reader)
constexpr uint32_t CB_ON = REC_CB0 + 13;    // bf16 [1,V] gated-normed o row -> o_proj input
constexpr uint32_t CB_SCR = REC_CB_X0 + 0;  // fp32 1 page, reader scratch: q / k / g face rows before the scatter
constexpr uint32_t CB_W = REC_CB_X0 + 1;    // fp32 [1,V] o_norm weight row tiles (read once, kept)
constexpr uint32_t CB_OT = REC_CB_X0 + 2;   // bf16 [B<=32, V] writer scratch: this head's o_proj input tiles
constexpr uint32_t CB_TAP = REC_CB_X0 + 3;  // fp32 [1,V] debug tap (REC_DEBUG): the pre-norm o row
constexpr uint32_t CB_PSUM = REC_CB_PSUM;
constexpr uint32_t CB_PIN = REC_CB_PIN;  // pages (b % 2) * (REC_SPREAD - 1) + slot: item b's partial from peer slot

// Stage 14c v2 (REC_V2, kda_rec_v2_sfpu.hpp): CB_Q carries one compact fp32 vector tile per item (VEC; CB_K / V / G /
// BETA / GO / KFULL / DCOL / O are not allocated); compute scratch: CB_VEC2 (VEC after prep) and CB_DELTA (o + the
// delta patterns), both packed and re-unpacked by compute; CB_W holds the o_norm weight in the u-layout (1 page)
constexpr uint32_t CB_VEC2 = CB_QCOL;
constexpr uint32_t CB_DELTA = CB_KCOL;
// VEC planes written by the reader (kda_rec_v2_sfpu.hpp kdav2::VQ ...)
constexpr uint32_t V2_Q = 0, V2_K = 4, V2_G = 8, V2_V = 12, V2_GO = 16, V2_BETA = 20;

// element offset (fp32 words, face-major tile) of DST plane p, lane t (mhc_pre_common.hpp plane_off: plane p = face
// p / 8, rows 4 ((p % 8) / 2) + t / 8, column 2 (t % 8) + p % 2 of that face)
constexpr uint32_t plane_off(uint32_t p, uint32_t t) {
    return (p >> 3) * 256u + (((p & 7u) >> 1) * 4u + (t >> 3)) * 16u + 2u * (t & 7u) + (p & 1u);
}
// compact key-side position of value r (0..127): plane r / 32, lane 8 (r % 4) + (r / 4) % 8
constexpr uint32_t key_off(uint32_t base, uint32_t r) {
    return plane_off(base + (r >> 5), 8u * (r & 3u) + ((r >> 2) & 7u));
}
// u-layout position of value j (0..127): plane j / 32, sub-row cg = 2 ((j / 16) % 2) + j % 2, column (j / 2) % 8
constexpr uint32_t val_off(uint32_t base, uint32_t j) {
    return plane_off(base + (j >> 5), 8u * (2u * ((j >> 4) & 1u) + (j & 1u)) + ((j >> 1) & 7u));
}
}  // namespace kda_rec
