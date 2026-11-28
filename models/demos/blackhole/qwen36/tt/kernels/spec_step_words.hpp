// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// The integer words of the two small programs around a Qwen3.6 MTP verify, shared by kernels/step_accept.cpp and
// kernels/chain_inputs.cpp:
//   accept_words: the step program (tt/spec_accept.py graph): the greedy walk over the verify ids and the drafter pack;
//   chain_words:  the chain program (tt/spec_chain.py chain_graph): the verify step buffer, the GDN verify ctrl pages
//                 and the sampler seeds of a chained verify.
// Plain C++ over word arrays: tests/test_spec_kernels_host.py compiles this header on the host and compares it with
// those graphs.
//
// Exactness. The graphs compute in float32 on integers whose magnitude stays below 2^24 (token ids, positions, table
// rows, page indices, device seeds, 0 / 1 flags). Float32 addition, subtraction, multiplication, comparison, min and
// max are exact there, so the same expressions in int32 give the same values; this file keeps each expression of the
// graphs as it is (a flag multiplies, it does not select), so a flag outside {0, 1} also gives the graph's value. The
// graphs end in ttnn.typecast: float32 to int32 keeps an integer, float32 to uint32 writes 0 for a negative value
// (to_u32); in-contract words never give a negative uint32 output.

#pragma once

#include <stdint.h>

#include "api/alignment.h"

namespace qwen36_spec {

// tt/spec_step.py rows of the verify step buffer (one word per verify row, then one word per user)
constexpr uint32_t VERIFY_TOKENS = 0, VERIFY_POSITIONS = 1, VERIFY_ROPE = 2;
constexpr uint32_t STEP_NUM_VALID = 3, STEP_DRAWN = 4, STEP_P0 = 5, STEP_ROPE_BASE = 6, STEP_DECLINED = 7;
constexpr uint32_t STEP_KEEP = 8, VERIFY_STEP_ROWS = 9;
constexpr uint32_t STEP_USER_ROWS = VERIFY_STEP_ROWS - STEP_NUM_VALID;  // STEP_NUM_VALID .. STEP_KEEP
// tt/spec_step.py drafter pack rows: 0 pending, 1 + k draft step k, K + 1 / K + 2 reseed tokens / positions,
// K + 3 anchor, K + 4 reseed RoPE rows, K + 5 + k RoPE row of draft step k
constexpr uint32_t pack_rows(uint32_t K) { return 2 * K + 5; }
constexpr uint32_t draft_anchor(uint32_t K) { return K + 3; }
// tt/spec_chain.py overlay rows (CH_*), one word per user
constexpr uint32_t CH_P0_PREV = 0, CH_ACTIVE = 1, CH_DELTA = 2, CH_NUM_VALID = 3, CH_DECLINED = 4, CH_KEEP = 5;
constexpr uint32_t CH_DRAWN_PREV = 6, CH_DRAWN = 7, CH_SRC = 8, CHAIN_OVERLAY_ROWS = 9;

// Every L1 slot starts on this boundary: a NOC read needs the L1 and DRAM addresses equal modulo the DRAM read
// alignment, 64 B on Blackhole (32 B on Wormhole), and every DRAM page (and column chunk) starts on it.
constexpr uint32_t SLOT = 64;
constexpr uint32_t slot_bytes(uint32_t words) { return align(words * 4u, SLOT); }

inline uint32_t to_u32(int32_t v) { return v < 0 ? 0u : static_cast<uint32_t>(v); }
inline int32_t min_i(int32_t a, int32_t b) { return a < b ? a : b; }
inline int32_t relu_i(int32_t a) { return a < 0 ? 0 : a; }
inline int32_t s32(uint32_t w) { return static_cast<int32_t>(w); }

// The verify id of row r after the draws merge: ids + drawn * (draws - ids); draws nullptr without a device sampler.
inline int32_t merged_id(const uint32_t* ids, const uint32_t* draws, int32_t drawn, uint32_t r) {
    const int32_t id = s32(ids[r]);
    return draws == nullptr ? id : id + drawn * (s32(draws[r]) - id);
}

// ------------------------------------------------------------------------------------------------ step program

struct AcceptShape {
    uint32_t B, K;
    uint32_t pack_words;  // words per drafter pack row, all written
    int32_t top;          // last RoPE table row
};

struct AcceptIn {
    const uint32_t* ids;        // verify argmax lanes, word u * T + j
    const uint32_t* draws;      // sampler draws (word u * T + j) or nullptr
    const uint32_t* tokens;     // step buffer row VERIFY_TOKENS
    const uint32_t* user_rows;  // step buffer rows STEP_NUM_VALID .. STEP_KEEP, B words each, user_stride apart
    uint32_t user_stride;
};

struct AcceptOut {
    uint32_t* pack;  // pack_rows(K) rows, pack_stride words apart
    uint32_t pack_stride;
    uint32_t* real;  // [B * T]: 1 where reseed row u * T + j keeps the user's page table row, else 0
};

// spec_accept.graph: m = the leading valid drafts equal to the id of the column before them; the drafter pack of the
// commit ids[u, 0 .. m]; the reseed rows j < m + declined * keep.
inline void accept_words(const AcceptShape& s, const AcceptIn& in, const AcceptOut& out) {
    const uint32_t B = s.B, K = s.K, T = K + 1;
    for (uint32_t i = 0; i < pack_rows(K); ++i) {
        for (uint32_t w = 0; w < s.pack_words; ++w) {
            out.pack[i * out.pack_stride + w] = 0;
        }
    }
    auto user = [&](uint32_t row, uint32_t u) {
        return s32(in.user_rows[(row - STEP_NUM_VALID) * in.user_stride + u]);
    };
    auto pack = [&](uint32_t row, uint32_t w) -> uint32_t& { return out.pack[row * out.pack_stride + w]; };
    for (uint32_t u = 0; u < B; ++u) {
        const int32_t nv = user(STEP_NUM_VALID, u), drawn = user(STEP_DRAWN, u), p0 = user(STEP_P0, u);
        const int32_t rb = user(STEP_ROPE_BASE, u), dec = user(STEP_DECLINED, u), keep = user(STEP_KEEP, u);
        auto id = [&](uint32_t j) { return merged_id(in.ids, in.draws, drawn, u * T + j); };
        uint32_t m = 0;
        while (m < K && id(m) == s32(in.tokens[u * T + m + 1]) && static_cast<int32_t>(m) < nv) {
            ++m;
        }
        const int32_t mi = static_cast<int32_t>(m);
        const int32_t drafting = 1 - dec;
        const int32_t kv = mi + dec * keep;
        pack(0, u) = to_u32(id(m) * drafting);
        const int32_t p = (p0 + mi) * drafting;
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t ki = static_cast<int32_t>(k);
            pack(1 + k, u) = to_u32(p + ki);
            pack(K + 5 + k, u) = to_u32(min_i(rb + mi + ki, s.top));
        }
        pack(draft_anchor(K), u) = u * T + m;
        for (uint32_t j = 0; j < T; ++j) {
            const int32_t ji = static_cast<int32_t>(j);
            const int32_t real = ji < kv ? 1 : 0;
            const uint32_t r = u * T + j;
            pack(K + 1, r) = to_u32(id(j) * real);
            pack(K + 2, r) = to_u32((p0 + ji) * real);
            pack(K + 4, r) = to_u32(min_i(rb + ji, s.top));
            out.real[r] = static_cast<uint32_t>(real);
        }
    }
}

// L1 bytes of kernels/step_accept.cpp (the same sum as spec_accept.step_kernel_l1_bytes): ids, draws, the step
// buffer's token row and user rows, the pack rows, the real-row flags, one scratch chunk and R page-table chunks.
struct AcceptLayout {
    uint32_t ids, draws, tokens, user_rows, pack, real, scratch, pt, bytes;
    constexpr AcceptLayout(uint32_t B, uint32_t K, uint32_t pack_words, uint32_t chunk) :
        ids(0),
        draws(ids + slot_bytes(B * (K + 1))),
        tokens(draws + slot_bytes(B * (K + 1))),
        user_rows(tokens + slot_bytes(B * (K + 1))),
        pack(user_rows + STEP_USER_ROWS * slot_bytes(B)),
        real(pack + pack_rows(K) * slot_bytes(pack_words)),
        scratch(real + slot_bytes(B * (K + 1))),
        pt(scratch + chunk),
        bytes(pt + B * (K + 1) * chunk) {}
};

// ----------------------------------------------------------------------------------------------- chain program

struct ChainShape {
    uint32_t B, K;
    uint32_t nv_heads, groups;  // GDN value heads per device, MambaSpec table groups (one ctrl page each)
    uint32_t step_words;        // words per step buffer row, all written
    uint32_t ctrl_words;        // words per ctrl page, all written
    int32_t top;                // last RoPE table row
};

struct ChainIn {
    const uint32_t* ids;     // verify argmax lanes of the predecessor, word src * T + j
    const uint32_t* draws;   // the predecessor's sampler draws, or nullptr without a device sampler
    const uint32_t* anchor;  // drafter pack row draft_anchor(K): src * T + m[src]
    const int32_t* overlay;  // CHAIN_OVERLAY_ROWS rows of B words, overlay_stride apart
    uint32_t overlay_stride;
    const uint32_t* drafts;  // draft k of predecessor row src at word (k * B + src) * draft_stride
    uint32_t draft_stride;
    const int32_t* cand;  // ctrl candidates: row c * B + u, 2 * nv_heads * groups words, cand_stride apart
    uint32_t cand_stride;
    const int32_t* seed_cand;  // seed candidates: row c * B + u, T words, seed_stride apart; nullptr without a sampler
    uint32_t seed_stride;
};

struct ChainOut {
    uint32_t* step;  // VERIFY_STEP_ROWS rows, step_stride words apart
    uint32_t step_stride;
    int32_t* ctrl;  // groups pages, ctrl_stride words apart
    uint32_t ctrl_stride;
    uint32_t* seeds;  // B * T words, or nullptr without a sampler
};

// spec_chain.chain_graph: row u of the next verify continues the predecessor's row src = overlay CH_SRC[u]. With m the
// predecessor's accepted drafts on row src (anchor[src] - src * T) and c = m + 1 its count, row u's tokens
// [ids[src, m], drafts of src...] (a draft column at or past num_valid repeats column 0), positions p0_prev + c + j,
// RoPE rows, the acceptance rows, and the ctrl words and seeds of candidate row m * B + u; every overlay word and
// candidate row is staged for row u. A src outside [0, B) reads 0 for its anchor, ids, draws and drafts, as the
// graph's one-hot gather gives. An m outside [0, T) picks nothing: column 0, the ctrl words and the seeds are 0, as
// the graph's one-hot sum gives.
inline void chain_words(const ChainShape& s, const ChainIn& in, const ChainOut& out) {
    const uint32_t B = s.B, K = s.K, T = K + 1, NV = s.nv_heads;
    for (uint32_t i = 0; i < VERIFY_STEP_ROWS; ++i) {
        for (uint32_t w = 0; w < s.step_words; ++w) {
            out.step[i * out.step_stride + w] = 0;
        }
    }
    for (uint32_t g = 0; g < s.groups; ++g) {
        for (uint32_t w = 0; w < s.ctrl_words; ++w) {
            out.ctrl[g * out.ctrl_stride + w] = 0;
        }
    }
    auto ovr = [&](uint32_t row, uint32_t u) { return in.overlay[row * in.overlay_stride + u]; };
    auto step = [&](uint32_t row, uint32_t w) -> uint32_t& { return out.step[row * out.step_stride + w]; };
    for (uint32_t u = 0; u < B; ++u) {
        const int32_t src = ovr(CH_SRC, u);
        const bool known = src >= 0 && src < static_cast<int32_t>(B);
        const uint32_t pred_row = known ? static_cast<uint32_t>(src) : 0u;
        const int32_t m = known ? s32(in.anchor[pred_row]) - static_cast<int32_t>(pred_row * T) : 0;
        const bool picked = m >= 0 && m < static_cast<int32_t>(T);
        const int32_t p0_prev = ovr(CH_P0_PREV, u), act = ovr(CH_ACTIVE, u), delta = ovr(CH_DELTA, u);
        const int32_t nv = ovr(CH_NUM_VALID, u), drawn_prev = ovr(CH_DRAWN_PREV, u);
        const int32_t tok0 =
            picked && known ? merged_id(in.ids, in.draws, drawn_prev, pred_row * T + static_cast<uint32_t>(m)) : 0;
        const int32_t p0 = (p0_prev + (m + 1)) * act;
        for (uint32_t j = 0; j < T; ++j) {
            const int32_t ji = static_cast<int32_t>(j);
            int32_t tok = tok0;
            if (j > 0 && ji - 1 < nv) {
                tok = known ? s32(in.drafts[((j - 1) * B + pred_row) * in.draft_stride]) : 0;
            }
            const uint32_t r = u * T + j;
            step(VERIFY_TOKENS, r) = to_u32(tok * act);
            step(VERIFY_POSITIONS, r) = to_u32(p0 + ji);
            step(VERIFY_ROPE, r) = to_u32(min_i(relu_i(p0 + ji + delta), s.top));
        }
        step(STEP_NUM_VALID, u) = to_u32(nv);
        step(STEP_DRAWN, u) = to_u32(ovr(CH_DRAWN, u));
        step(STEP_P0, u) = to_u32(p0);
        step(STEP_ROPE_BASE, u) = to_u32(min_i(relu_i(p0 + delta), s.top));
        step(STEP_DECLINED, u) = to_u32(ovr(CH_DECLINED, u));
        step(STEP_KEEP, u) = to_u32(ovr(CH_KEEP, u));
        const int32_t* cand = picked ? in.cand + (static_cast<uint32_t>(m) * B + u) * in.cand_stride : nullptr;
        for (uint32_t g = 0; g < s.groups; ++g) {
            int32_t* page = out.ctrl + g * out.ctrl_stride;
            for (uint32_t h = 0; h < NV; ++h) {
                page[1 + B + u * NV + h] = cand == nullptr ? 0 : cand[2 * NV * g + h];
                page[1 + B + B * NV + u * NV + h] = cand == nullptr ? 0 : cand[2 * NV * g + NV + h];
            }
        }
        if (out.seeds != nullptr) {
            const int32_t* seed = picked ? in.seed_cand + (static_cast<uint32_t>(m) * B + u) * in.seed_stride : nullptr;
            for (uint32_t j = 0; j < T; ++j) {
                out.seeds[u * T + j] = seed == nullptr ? 0u : to_u32(seed[j]);
            }
        }
    }
}

// L1 bytes of kernels/chain_inputs.cpp (the same sum as spec_chain.chain_kernel_l1_bytes): ids, draws, the anchor
// row, the overlay rows, one slot per draft id, the ctrl and seed candidates, then the step rows, ctrl pages and seeds.
struct ChainLayout {
    uint32_t ids, draws, anchor, overlay, drafts, cand, seed_cand, step, ctrl, seeds, bytes;
    constexpr ChainLayout(
        uint32_t B,
        uint32_t K,
        uint32_t nv_heads,
        uint32_t groups,
        uint32_t step_words,
        uint32_t ctrl_words,
        bool sampler) :
        ids(0),
        draws(ids + slot_bytes(B * (K + 1))),
        anchor(draws + slot_bytes(B * (K + 1))),
        overlay(anchor + slot_bytes(B)),
        drafts(overlay + CHAIN_OVERLAY_ROWS * slot_bytes(B)),
        cand(drafts + K * B * SLOT),
        seed_cand(cand + (K + 1) * B * slot_bytes(2 * nv_heads * groups)),
        step(seed_cand + (sampler ? (K + 1) * B * slot_bytes(K + 1) : 0)),
        ctrl(step + VERIFY_STEP_ROWS * slot_bytes(step_words)),
        seeds(ctrl + groups * slot_bytes(ctrl_words)),
        bytes(seeds + (sampler ? slot_bytes(B * (K + 1)) : 0)) {}
};

}  // namespace qwen36_spec
