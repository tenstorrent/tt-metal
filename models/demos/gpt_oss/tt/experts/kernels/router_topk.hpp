// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Router top-k + softmax on a data-movement core (no FPU): top-k by order-preserving integer keys of the BF16
// logits (ties to the lower id), softmax over the selected logits in Q16 fixed point (exp2 = integer shift x
// degree-5 polynomial, ~1e-5 relative), rounded to BF16. Shared by the decode router stream kernels.

#pragma once

#include <stdint.h>

namespace router_topk {
// BF16 bits -> Q16.16 fixed point (|x| < 2^14).
inline int32_t bf16_to_q16(uint16_t b) {
    const int32_t e = (b >> 7) & 0xFF;
    if (e == 0) {
        return 0;
    }
    const int32_t m = (b & 0x7F) | 0x80;
    const int32_t shift = e - 118;  // value = m * 2^(e - 134); in Q16: m << (e - 118)
    int32_t v = shift >= 0 ? (shift > 22 ? (m << 22) : (m << shift)) : (shift < -31 ? 0 : (m >> -shift));
    return (b & 0x8000) ? -v : v;
}

// exp(d) in Q16 for d <= 0 in Q16.
inline uint32_t exp_q16(int32_t d) {
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d) * 94548) >> 16);  // d * log2(e), Q16
    const int32_t n = t >> 16;                                                        // floor
    const uint32_t f = static_cast<uint32_t>(t) & 0xFFFF;                             // [0, 1) in Q16
    uint32_t p = 87;                                                                  // 2^f, Horner, Q16
    p = 630 + ((p * f) >> 16);
    p = 3638 + ((p * f) >> 16);
    p = 15743 + ((p * f) >> 16);
    p = 45426 + ((p * f) >> 16);
    p = 65536 + ((p * f) >> 16);
    return n <= -31 ? 0 : p >> -n;
}

// Q16 in [0, 1] -> BF16 bits, round to nearest.
inline uint16_t q16_to_bf16(uint32_t w) {
    if (w == 0) {
        return 0;
    }
    int32_t msb = 31 - __builtin_clz(w);
    int32_t shift = msb - 7;
    uint32_t m = shift > 0 ? (w + (1u << (shift - 1))) >> shift : w << -shift;
    if (m >= 0x100) {  // rounding carried into a new bit
        m >>= 1;
        msb += 1;
    }
    return static_cast<uint16_t>(((127 + msb - 16) << 7) | (m & 0x7F));
}
}  // namespace router_topk

namespace router_topk {
// Writes 8 UINT16 ids then 8 BF16 scores (first topk valid, rest zero) to `stage` (32 bytes).
template <uint32_t topk, uint32_t n_logits>
inline void topk_softmax(volatile tt_l1_ptr uint16_t* logits, volatile tt_l1_ptr uint16_t* stage) {
    uint32_t keys[n_logits];
    for (uint32_t i = 0; i < n_logits; ++i) {
        const uint32_t b = logits[i];
        keys[i] = (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000);
    }
    uint32_t ids[topk];
    int32_t vals[topk];
    for (uint32_t j = 0; j < topk; ++j) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < n_logits; ++i) {
            best = keys[i] > keys[best] ? i : best;
        }
        ids[j] = best;
        vals[j] = bf16_to_q16(logits[best]);
        keys[best] = 0;
    }
    uint32_t e[topk];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < topk; ++j) {
        e[j] = exp_q16(vals[j] - vals[0]);
        sum += e[j];
    }
    for (uint32_t i = 0; i < 16; ++i) {
        stage[i] = 0;
    }
    for (uint32_t j = 0; j < topk; ++j) {
        stage[j] = static_cast<uint16_t>(ids[j]);
        stage[8 + j] = q16_to_bf16((e[j] << 15) / (sum >> 1));
    }
}
}  // namespace router_topk
