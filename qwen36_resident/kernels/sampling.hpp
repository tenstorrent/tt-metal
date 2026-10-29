// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// On-device sampling of the resident decode (batch 1): each streamer keeps the top kTopK of its logits, the hub
// merges them into the chip's top kTopK, the hubs exchange those over fabric and every hub merges the chips' lists
// and draws the same token from them (same lists, parameters and position on every chip).
//
// A list is kTopK order keys then kTopK vocab indices (kListBytes), largest first, the smaller index first on
// equal keys, so entry 0 is the greedy token of the argmax rule. Unused entries hold key 0 (below every logit).
//
// The draw is temperature, then top-k, then top-p (vLLM's order) over the global list, in Q16 fixed point: the data
// movement cores have no FPU. Parameters (4 words): 1 / temperature in Q16 (0 = greedy), top-k (<= kTopK, <= 1 =
// greedy), top-p in Q16 (65536 = off), seed; the random number is a hash of (seed, position), so a seeded request
// repeats and nothing carries over between steps.
#pragma once

#include <stdint.h>

namespace sampling {

constexpr uint32_t kTopK = 32;
constexpr uint32_t kListBytes = 8 * kTopK;

// bf16 bits -> unsigned key in the order of the values, and back
inline uint32_t key_of(uint32_t b) { return (b & 0x8000) ? (~b & 0xffff) : (b | 0x8000); }
inline uint32_t bf16_of(uint32_t key) { return (key & 0x8000) ? (key & 0x7fff) : (~key & 0xffff); }

inline bool better(uint32_t k, uint32_t i, uint32_t bk, uint32_t bi) { return k > bk || (k == bk && i < bi); }

// insert (key, index) into a list (keys, then indices kTopK words on) if it ranks in it; indices arrive in
// ascending order, so an equal key ranks after the ones already there
inline void insert(volatile tt_l1_ptr uint32_t* list, uint32_t key, uint32_t index) {
    volatile tt_l1_ptr uint32_t* idx = list + kTopK;
    if (key <= list[kTopK - 1]) {
        return;
    }
    uint32_t i = kTopK - 1;
    while (i > 0 && list[i - 1] < key) {
        list[i] = list[i - 1];
        idx[i] = idx[i - 1];
        i--;
    }
    list[i] = key;
    idx[i] = index;
}

inline void clear(volatile tt_l1_ptr uint32_t* list) {
    for (uint32_t i = 0; i < kTopK; i++) {
        list[i] = 0;
        list[kTopK + i] = 0xffffffff;
    }
}

// the top kTopK of n <= 32 lists at `lists` (kListBytes apart) into `out`
inline void merge(const volatile tt_l1_ptr uint32_t* lists, uint32_t n, volatile tt_l1_ptr uint32_t* out) {
    uint8_t head[32];
    for (uint32_t s = 0; s < n; s++) {
        head[s] = 0;
    }
    for (uint32_t r = 0; r < kTopK; r++) {
        uint32_t best = n, bk = 0, bi = 0;
        for (uint32_t s = 0; s < n; s++) {
            if (head[s] == kTopK) {
                continue;
            }
            const volatile tt_l1_ptr uint32_t* a = lists + s * 2 * kTopK;
            if (best == n || better(a[head[s]], a[kTopK + head[s]], bk, bi)) {
                best = s;
                bk = a[head[s]];
                bi = a[kTopK + head[s]];
            }
        }
        out[r] = bk;
        out[kTopK + r] = bi;
        head[best]++;
    }
}

// bf16 bits -> Q16.16 (|x| < 2^14)
inline int32_t q16(uint32_t b) {
    const int32_t e = (b >> 7) & 0xff;
    if (e == 0) {
        return 0;
    }
    const int32_t m = (b & 0x7f) | 0x80;
    const int32_t shift = e - 118;  // value = m * 2^(e - 134), in Q16 m << (e - 118)
    const int32_t v = shift >= 0 ? (shift > 22 ? (m << 22) : (m << shift)) : (shift < -31 ? 0 : (m >> -shift));
    return (b & 0x8000) ? -v : v;
}

// exp(d) in Q16 for d <= 0 in Q16: 2^(d log2 e), the fraction by a degree-5 polynomial (~1e-5 relative)
inline uint32_t exp_q16(int32_t d) {
    const int32_t t = static_cast<int32_t>((static_cast<int64_t>(d) * 94548) >> 16);
    const int32_t n = t >> 16;
    const uint32_t f = static_cast<uint32_t>(t) & 0xffff;
    uint32_t p = 87;
    p = 630 + ((p * f) >> 16);
    p = 3638 + ((p * f) >> 16);
    p = 15743 + ((p * f) >> 16);
    p = 45426 + ((p * f) >> 16);
    p = 65536 + ((p * f) >> 16);
    return n <= -31 ? 0 : p >> -n;
}

inline uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

// the token drawn from the global list; w: kTopK words of scratch
inline uint32_t draw(
    const volatile tt_l1_ptr uint32_t* list,
    const volatile tt_l1_ptr uint32_t* params,
    uint32_t pos,
    volatile tt_l1_ptr uint32_t* w) {
    const uint32_t inv_temp = params[0];
    const uint32_t k = params[1] < kTopK ? params[1] : kTopK;
    if (inv_temp == 0 || k <= 1) {
        return list[kTopK];
    }
    const int32_t l0 = q16(bf16_of(list[0]));
    uint32_t total = 0;
    for (uint32_t i = 0; i < k; i++) {
        int64_t d = (static_cast<int64_t>(q16(bf16_of(list[i])) - l0) * inv_temp) >> 16;
        d = d < -(64 << 16) ? -(64 << 16) : d;
        w[i] = list[i] ? exp_q16(static_cast<int32_t>(d)) : 0;
        total += w[i];
    }
    // top-p: the tokens whose higher-ranked mass is below top_p of the total (at least the first)
    const uint32_t limit = static_cast<uint32_t>((static_cast<uint64_t>(total) * params[2]) >> 16);
    uint32_t kept = 0, mass = 0;
    while (kept < k && (kept == 0 || mass < limit)) {
        mass += w[kept++];
    }
    uint32_t target =
        static_cast<uint32_t>((static_cast<uint64_t>(mix(params[3] ^ mix(pos + 0x9e3779b9))) * mass) >> 32);
    for (uint32_t i = 0; i < kept; i++) {
        if (target < w[i]) {
            return list[kTopK + i];
        }
        target -= w[i];
    }
    return list[kTopK + kept - 1];
}

}  // namespace sampling
