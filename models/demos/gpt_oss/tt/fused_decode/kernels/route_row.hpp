// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// One row's routing as TopKRouter.dense_decode does it, on a RISC-V core (no FPU), shared by the grouped MoE's route
// program (gmoe_route.cpp) and the batched decode's router (dense_route.cpp): the routed experts are every logit >= the
// K_SEL-th largest (with multiplicity, so a BF16 tie routes more), at most K, in descending logit order, then ascending
// index; their softmax weights in integer fixed point (exp(d) = 2^(d log2 e) by a Q30 polynomial, one 64-bit
// reciprocal per row), rounded to BF16.

#pragma once

#include <stdint.h>

namespace route_row {
// BF16 bits -> an unsigned key in the same order as the values
inline uint32_t key16(uint32_t b) { return (b & 0x8000) ? (~b & 0xFFFF) : (b | 0x8000); }
// BF16 bits -> signed Q16 fixed point (|value| < 2^15)
inline int32_t q16(uint32_t b) {
    const int32_t e = static_cast<int32_t>((b >> 7) & 0xFF);
    if (e == 0) {
        return 0;
    }
    const int32_t m = static_cast<int32_t>(0x80 | (b & 0x7F));
    const int32_t sh = e - 118;  // value = m 2^(e - 127 - 7); Q16: << 16
    int32_t v = sh >= 0 ? (sh < 23 ? (m << sh) : 0x7FFFFFFF) : (sh > -31 ? (m >> -sh) : 0);
    return (b & 0x8000) ? -v : v;
}
// 2^(t / 2^16) for t <= 0 as Q30 (0 below 2^-30)
inline uint32_t exp2_q30(int64_t t) {
    const int32_t n = static_cast<int32_t>(t >> 16);  // floor
    if (n < -30) {
        return 0;
    }
    const uint32_t f = static_cast<uint32_t>(t - (static_cast<int64_t>(n) << 16));  // [0, 2^16)
    // 2^x on [0, 1): Taylor to degree 7 in Horner form, Q30 (max relative error ~1e-6)
    const int64_t x = static_cast<int64_t>(f) << 14;  // Q30
    const int64_t c[7] = {744261118, 257941248, 59597083, 10327387, 1431680, 165394, 16377};
    int64_t p = c[6];
    for (int32_t k = 5; k >= 0; --k) {
        p = c[k] + ((p * x) >> 30);
    }
    p = (1LL << 30) + ((p * x) >> 30);  // [2^30, 2^31)
    return static_cast<uint32_t>(p >> (-n));
}
// q / 2^30 (0 < q <= 2^30) -> BF16 bits, round to nearest even
inline uint16_t bf16_q30(uint32_t q) {
    if (q == 0) {
        return 0;
    }
    const int32_t msb = 31 - __builtin_clz(q);
    int32_t e = msb - 30 + 127;
    uint32_t m;
    const int32_t sh = msb - 7;
    if (sh > 0) {
        m = q >> sh;
        const uint32_t rem = q & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1))) {
            ++m;
        }
        if (m == 256) {
            m = 128;
            ++e;
        }
    } else {
        m = q << (-sh);
    }
    return static_cast<uint16_t>((e << 7) | (m & 0x7F));
}

// b: the row's E logits (BF16 bits) -> sel[0 .. n - 1] the routed experts, taken (bit e: routed), wb their BF16 weights
template <uint32_t E, uint32_t K, uint32_t K_SEL>
inline void route(const uint32_t* b, uint32_t* sel, uint32_t& n, uint32_t& taken, uint16_t* wb) {
    uint32_t key[E];
    for (uint32_t e = 0; e < E; ++e) {
        key[e] = key16(b[e]);
    }
    // the K_SEL largest (key, index) by insertion (an equal key goes after the ones already in: the lower index
    // first), then every other expert tied with the K_SEL-th in ascending index order, at most K in all: the same
    // routed set and order as keeping the K largest, at K_SEL-wide insertion cost
    uint32_t tk[K_SEL], ti[K_SEL], m = 0;
    for (uint32_t e = 0; e < E; ++e) {
        const uint32_t v = key[e];
        if (m == K_SEL && v <= tk[K_SEL - 1]) {
            continue;
        }
        uint32_t p = m < K_SEL ? m : K_SEL - 1;
        while (p > 0 && tk[p - 1] < v) {
            tk[p] = tk[p - 1];
            ti[p] = ti[p - 1];
            --p;
        }
        tk[p] = v;
        ti[p] = e;
        if (m < K_SEL) {
            ++m;
        }
    }
    const uint32_t kth = tk[K_SEL - 1];
    n = 0;
    taken = 0;
    for (; n < m; ++n) {
        sel[n] = ti[n];
        taken |= 1u << ti[n];
    }
    for (uint32_t e = 0; e < E && n < K; ++e) {
        if (key[e] == kth && !((taken >> e) & 1u)) {
            sel[n++] = e;
            taken |= 1u << e;
        }
    }
    // softmax over the selected, fixed point: e_k = 2^((l_k - max) log2 e) (Q30), w_k = e_k / sum
    const int32_t qmax = q16(b[sel[0]]);
    uint32_t ex[K];
    uint64_t sum = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const int64_t d = static_cast<int64_t>(q16(b[sel[k]])) - qmax;  // <= 0, Q16
        ex[k] = exp2_q30((d * 94548) >> 16);                            // log2 e = 94548 / 2^16
        sum += ex[k];
    }
    // one 64-bit division per row (the RISC-V cores divide in software): inv = 2^62 / sum, w = ex inv / 2^32
    const uint64_t inv = (1ULL << 62) / sum;
    for (uint32_t k = 0; k < n; ++k) {
        wb[k] = bf16_q30(static_cast<uint32_t>((static_cast<uint64_t>(ex[k]) * inv) >> 32));
    }
}
}  // namespace route_row
