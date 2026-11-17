// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Batch-1 router tail math (data-movement cores, integer only): top-k of the FP32 router logits, softmax over the k
// picks (equal to softmax over all E, then top-k, renormalized), times per_expert_scale[id], as the packed routing row
// RouterTopk.op_packed produces: [w_0 .. w_{k-1} | id_0 .. id_{k-1}] bf16 (ids exact below 256). Order: descending
// logit; equal logits keep the lower expert id first. Used by router_topk_b1.cpp and fused_decode
// router_red_writer.cpp.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

namespace router_tail {

// The cores have no FPU: everything below is integer arithmetic.

// order-preserving map of IEEE float bits to unsigned integers
inline uint32_t sortable(uint32_t bits) { return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u); }

// IEEE float bits -> signed fixed point with 16 fractional bits (|value| < 2^15; smaller than 2^-16 -> 0)
inline int64_t to_q16(uint32_t bits) {
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xFF);
    if (exponent == 0) {
        return 0;
    }
    const int64_t mantissa = static_cast<int64_t>((bits & 0x7FFFFFu) | 0x800000u);  // value = m * 2^(e - 150)
    const int32_t shift = exponent - 150 + 16;
    int64_t q = shift >= 0 ? (mantissa << shift) : (shift > -40 ? (mantissa >> -shift) : 0);
    return (bits & 0x80000000u) ? -q : q;
}

// 2^y for y <= 0, y in Q16 -> Q30 (2^r on r in [0, 1) by a degree-5 polynomial in Q30, rel. error ~3e-7)
inline int64_t exp2_q30(int64_t y) {
    const int64_t n = y >> 16;        // floor
    const int64_t r = y - (n << 16);  // [0, 2^16)
    if (n <= -31) {
        return 0;
    }
    const int64_t rq = r << 14;  // Q30
    constexpr int64_t c[6] = {1073741824LL, 744261117LL, 257941803LL, 59603845LL, 10351951LL, 1355011LL};
    int64_t p = c[5];
    for (int i = 4; i >= 0; --i) {
        p = ((p * rq) >> 30) + c[i];
    }
    return p >> static_cast<int32_t>(-n);
}

// positive Q16 value -> bf16 bits, round to nearest even
inline uint16_t q16_to_bf16(int64_t q) {
    if (q <= 0) {
        return 0;
    }
    // 64-bit leading-zero count from two 32-bit ones (RV32: __builtin_clzll is a libgcc call)
    const uint32_t hi = static_cast<uint32_t>(static_cast<uint64_t>(q) >> 32);
    const int32_t msb = hi ? 63 - __builtin_clz(hi) : 31 - __builtin_clz(static_cast<uint32_t>(q));
    const uint32_t exponent = static_cast<uint32_t>(msb - 16 + 127);
    // 24-bit mantissa with the hidden bit, then round to bf16's 8
    const uint64_t m24 = msb >= 23 ? static_cast<uint64_t>(q) >> (msb - 23) : static_cast<uint64_t>(q) << (23 - msb);
    const uint32_t bits = (exponent << 23) | static_cast<uint32_t>(m24 & 0x7FFFFFu);
    return static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

inline uint16_t uint_to_bf16(uint32_t v) { return q16_to_bf16(static_cast<int64_t>(v) << 16); }

// logit: E FP32 bit patterns; pes: E bf16 per-expert scales; row: 2 k bf16 outputs
template <uint32_t E, uint32_t k, typename Logit>
inline void topk_softmax_at(Logit logit, volatile tt_l1_ptr uint16_t* pes, volatile tt_l1_ptr uint16_t* row) {
    // one pass: keep the k largest (key, id) sorted descending; strict > keeps the lower id first on ties
    uint32_t key[k], id[k];
    for (uint32_t i = 0; i < k; ++i) {
        key[i] = 0;
        id[i] = 0;
    }
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t v = sortable(logit(e));
        if (v <= key[k - 1]) {
            continue;
        }
        uint32_t pos = k - 1;
        while (pos > 0 && v > key[pos - 1]) {
            key[pos] = key[pos - 1];
            id[pos] = id[pos - 1];
            --pos;
        }
        key[pos] = v;
        id[pos] = e;
    }
    // softmax over the picks relative to the largest (Q16 logit differences, Q30 exponentials), times the
    // per-expert scale (bf16 -> Q16), normalized: Q16 weights
    constexpr int64_t log2e_q30 = 1549082005LL;  // log2(e) in Q30
    const int64_t top = to_q16(logit(id[0]));
    int64_t e[k];
    int64_t sum = 0;
    for (uint32_t i = 0; i < k; ++i) {
        e[i] = exp2_q30(((to_q16(logit(id[i])) - top) * log2e_q30) >> 30);
        sum += e[i];
    }
    // 1 / sum with one 32-bit (hardware) division: the Q30 sum (<= k * 2^30) as Q16 fits 32 bits, inv = 2^32 / sum16
    const uint32_t sum16 = static_cast<uint32_t>(sum >> 14);
    const uint32_t inv = 0xFFFFFFFFu / sum16;
    for (uint32_t i = 0; i < k; ++i) {
        const int64_t scale = to_q16(static_cast<uint32_t>(pes[id[i]]) << 16);
        const uint64_t share = (static_cast<uint64_t>(static_cast<uint32_t>(e[i] >> 14)) * inv) >> 16;  // Q16
        row[i] = q16_to_bf16((static_cast<int64_t>(share) * scale) >> 16);
        row[k + i] = uint_to_bf16(id[i]);
    }
}

// logit: E FP32 bit patterns, contiguous
template <uint32_t E, uint32_t k>
inline void topk_softmax(
    volatile tt_l1_ptr uint32_t* logit, volatile tt_l1_ptr uint16_t* pes, volatile tt_l1_ptr uint16_t* row) {
    topk_softmax_at<E, k>([logit](uint32_t e) -> uint32_t { return logit[e]; }, pes, row);
}

// logits: row 0 of E / 32 FP32 tiles (face 0 row 0 = columns 0..15, face 1 row 0 = 16..31), read in place
template <uint32_t E, uint32_t k>
inline void topk_softmax_tiles(
    volatile tt_l1_ptr uint32_t* tiles, volatile tt_l1_ptr uint16_t* pes, volatile tt_l1_ptr uint16_t* row) {
    topk_softmax_at<E, k>(
        [tiles](uint32_t e) -> uint32_t {
            const uint32_t c = e & 31;
            return tiles[1024 * (e >> 5) + (c < 16 ? c : 240 + c)];
        },
        pes,
        row);
}

}  // namespace router_tail
