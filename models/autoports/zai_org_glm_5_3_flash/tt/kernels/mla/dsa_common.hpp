// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 7b DSA decode (5d m4_phase_spec I4 / I5 / S1) on the attention owners, per token t at position p:
//   sparse  <=>  kv_len = p + 1 > 2051 (n_closed = (p + 1) / 4 > 512 complete pools; below that every token is
//                selected and the dense phase D1 is exact);
//   I4 score: owner (g, c) scores the pooled-key tiles pt = c, c + C, ... < ceil(n_closed / 32) of its slot:
//             S = q_idx (32 index heads x 128) . K_pool^T -> relu -> wsc (rows = tokens, cols = heads; pre-scaled) @ S
//             -> row t = the 32 pools' fp32 scores (the NCRISC keeps the pools < n_closed only; every owner of a
//             chunk computes the same scores, group 0 reports them);
//   I5 select: exact top-512 by a distributed radix select over the fp32 score keys (4-bit digits, 8 rounds + one
//             tie round), group-0 owners report 16-bin histograms to the coordinator (the index core's BRISC), which
//             broadcasts (prefix, k_remaining) per round and per-core tie quotas (core order) at the end;
//   S1 attend: each owner attends its own selected pools (4 latent rows each, gathered from the paged cache as 128-B
//             face-row runs) + core 0 the tail pool (the < 4 rows of the incomplete pool, always selected), in
//             batches of <= 3 K tiles combined locally (online softmax), then the group's partial tree as D1.
#pragma once

#include <cstdint>

namespace mladsa {
constexpr uint32_t DENSE_MAX = 2051;
constexpr uint32_t TOPK = 512;
constexpr uint32_t KPOOL = 4;
constexpr uint32_t DIGIT_BITS = 4, BINS = 1u << DIGIT_BITS, ROUNDS = 32 / DIGIT_BITS;
constexpr uint32_t MSGS = ROUNDS + 1;  // + the tie round
constexpr uint32_t MSG_BYTES = 16;     // mailbox slot: prefix, k_rem / quota
constexpr uint32_t HIST_BYTES = 64;    // per-core histogram slot on the coordinator (16 x uint32)
// rows one launch serves (the mailbox): MLA_DSA_TMAX from tt/mk_mla.py (dsa_tmax(t): 4, or 32 at t > 4)
#if defined(MLA_DSA_TMAX)
constexpr uint32_t T_MAX = MLA_DSA_TMAX;
#else
constexpr uint32_t T_MAX = 4;
#endif
// cur_pos in L1 scratch: one DRAM page of <= 32 int32 ([t] or [1, 32]; a page read lands its 64-B-rounded size)
constexpr uint32_t POS_BYTES = 128;
constexpr uint32_t POOLS_PER_TILE = 8;
constexpr uint32_t SEL_MAX = 512;  // selected pools one core can hold (+ header)
constexpr uint32_t SEL_HDR = 3;    // CB_ASEL words: n_sel, tail rows (core 0), tail pool, then the n_sel pool ids
constexpr uint32_t MAX_CHUNKS = 64;
// I4 score batch (Stage 14 rev 37, tt/mk_mla.py GLM53_MLA_DSA_SB): pooled-key tiles read per barrier (BRISC) and scored
// per DST acquire (compute: SB fp32 DST slots; half sync holds 4)
#if defined(MLA_DSA_SB)
constexpr uint32_t SB = MLA_DSA_SB;
#else
constexpr uint32_t SB = 1;
#endif
static_assert(SB >= 1 && SB <= 4, "score batch: <= 4 fp32 DST tiles");
// the owners' mailbox (an L1 tensor sharded over the owners, the same address on every owner; written by the
// coordinator): per token t and round r a 16-B message (prefix, k_remaining) at (t MSGS + r) 16, then per token the
// tie-round quotas of the chunk cores (uint32 per chunk) at QUOTA_OFF + t MAX_CHUNKS 4
// (QUOTA_W words per chunk core: the quota; with MLA_DSA_SHARD also the core's candidate record offset,
// kernels/mla/dsa_shard.hpp)
#if defined(MLA_DSA_SHARD)
constexpr uint32_t QUOTA_W = 2;
#else
constexpr uint32_t QUOTA_W = 1;
#endif
constexpr uint32_t QUOTA_OFF = T_MAX * MSGS * MSG_BYTES;
constexpr uint32_t MBOX_BYTES = QUOTA_OFF + T_MAX * MAX_CHUNKS * 4 * QUOTA_W;
// the coordinator's histogram slots (an L1 tensor on the index core): chunk core c writes 16 uint32 at c 64
constexpr uint32_t HIST_SLOT_BYTES = HIST_BYTES;

inline bool sparse(uint32_t p) { return p + 1 > DENSE_MAX; }
// order-preserving key of an fp32 bit pattern
inline uint32_t key_of(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }
// pool id of local key index i of chunk core c (local tile i / 32 = pooled-key tile c + (i / 32) C)
inline uint32_t pool_of(uint32_t i, uint32_t c, uint32_t C) { return (c + (i / 32) * C) * 32 + (i % 32); }
// pooled-key tiles taken by chunk core c at n_closed complete pools
inline uint32_t pool_tiles(uint32_t n_closed, uint32_t c, uint32_t C) {
    const uint32_t nt = (n_closed + 31) / 32;
    return c < nt ? (nt - c + C - 1) / C : 0;
}
// rev 52: the I5 coordinator's histogram sum (C owners x BINS words) with the BINS accumulators in registers: one L1
// load per word. The rev <= 51 loop (h[b] += slot[b] on a stack array: 3 L1 accesses per word) cost ~0.23 us per owner
// per round -- 14.7 us of the ~15-us t = 1 radix round with 64 owners (desktop-0 d0f / card0 u58 probes).
inline __attribute__((always_inline)) void sum_hist(volatile tt_l1_ptr uint32_t* hist, uint32_t C, uint32_t* out) {
    static_assert(BINS == 16, "sum_hist: 16 accumulators");
    uint32_t a0 = 0;
    uint32_t a1 = 0;
    uint32_t a2 = 0;
    uint32_t a3 = 0;
    uint32_t a4 = 0;
    uint32_t a5 = 0;
    uint32_t a6 = 0;
    uint32_t a7 = 0;
    uint32_t a8 = 0;
    uint32_t a9 = 0;
    uint32_t a10 = 0;
    uint32_t a11 = 0;
    uint32_t a12 = 0;
    uint32_t a13 = 0;
    uint32_t a14 = 0;
    uint32_t a15 = 0;
    for (uint32_t c = 0; c < C; ++c) {
        volatile tt_l1_ptr uint32_t* p = hist + c * (HIST_SLOT_BYTES / 4);
        a0 += p[0];
        a1 += p[1];
        a2 += p[2];
        a3 += p[3];
        a4 += p[4];
        a5 += p[5];
        a6 += p[6];
        a7 += p[7];
        a8 += p[8];
        a9 += p[9];
        a10 += p[10];
        a11 += p[11];
        a12 += p[12];
        a13 += p[13];
        a14 += p[14];
        a15 += p[15];
    }
    out[0] = a0;
    out[1] = a1;
    out[2] = a2;
    out[3] = a3;
    out[4] = a4;
    out[5] = a5;
    out[6] = a6;
    out[7] = a7;
    out[8] = a8;
    out[9] = a9;
    out[10] = a10;
    out[11] = a11;
    out[12] = a12;
    out[13] = a13;
    out[14] = a14;
    out[15] = a15;
}
}  // namespace mladsa
