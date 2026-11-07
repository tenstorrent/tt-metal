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

namespace router_topk {
// topk_softmax for top-4 of 32 in one pass (ROUTER_FAST_TOPK): the logits read as 16 words, a register top-4 by order
// key (strict >, ascending ids: ties keep the lower id, as topk_softmax's repeated scans), the same Q16 softmax.
// Same output bits as topk_softmax<4, 32>.
inline void topk4_softmax_fast(volatile tt_l1_ptr uint16_t* logits, volatile tt_l1_ptr uint16_t* stage) {
    const volatile tt_l1_ptr uint32_t* w = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(logits);
    uint32_t k0 = 0, k1 = 0, k2 = 0, k3 = 0, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
    bool any = false;
    auto ins = [&](uint32_t b, uint32_t id) __attribute__((always_inline)) {
        const uint32_t key = (b & 0x8000) ? (~b & 0xFFFFu) : (b | 0x8000u);
        if (key > k3 || !any) {
            any = true;
            if (key > k2) {
                k3 = k2;
                i3 = i2;
                if (key > k1) {
                    k2 = k1;
                    i2 = i1;
                    if (key > k0) {
                        k1 = k0;
                        i1 = i0;
                        k0 = key;
                        i0 = id;
                    } else {
                        k1 = key;
                        i1 = id;
                    }
                } else {
                    k2 = key;
                    i2 = id;
                }
            } else {
                k3 = key;
                i3 = id;
            }
        }
    };
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t v = w[i];
        ins(v & 0xFFFFu, 2 * i);
        ins(v >> 16, 2 * i + 1);
    }
    auto raw = [](uint32_t key) -> uint16_t {
        return static_cast<uint16_t>((key & 0x8000u) ? (key & 0x7FFFu) : (~key & 0xFFFFu));
    };
    const uint32_t ids[4] = {i0, i1, i2, i3};
    const int32_t v0 = bf16_to_q16(raw(k0));
    const int32_t vals[4] = {v0, bf16_to_q16(raw(k1)), bf16_to_q16(raw(k2)), bf16_to_q16(raw(k3))};
    uint32_t e[4];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        e[j] = exp_q16(vals[j] - v0);
        sum += e[j];
    }
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    st[0] = ids[0] | (ids[1] << 16);
    st[1] = ids[2] | (ids[3] << 16);
    st[2] = 0;
    st[3] = 0;
    const uint32_t d = sum >> 1;
    st[4] = q16_to_bf16((e[0] << 15) / d) | (static_cast<uint32_t>(q16_to_bf16((e[1] << 15) / d)) << 16);
    st[5] = q16_to_bf16((e[2] << 15) / d) | (static_cast<uint32_t>(q16_to_bf16((e[3] << 15) / d)) << 16);
    st[6] = 0;
    st[7] = 0;
}
}  // namespace router_topk

namespace router_topk {
// Top-4 of more than 32 experts (ROUTER_MERGE: gpt-oss-120b's 128, one 32-expert logit column per router core). Merge
// entries: (1 << 31) | order key << 8 | (255 - expert id): larger = better, ties to the lower id (ids < 128).
inline void top4_insert(uint32_t e, uint32_t& t0, uint32_t& t1, uint32_t& t2, uint32_t& t3) {
    if (e > t3) {
        if (e > t2) {
            t3 = t2;
            if (e > t1) {
                t2 = t1;
                if (e > t0) {
                    t1 = t0;
                    t0 = e;
                } else {
                    t1 = e;
                }
            } else {
                t2 = e;
            }
        } else {
            t3 = e;
        }
    }
}

// This core's top-4 merge entries of its 32 logits (expert ids id0 .. id0 + 31) into top[0..3], best first.
inline void top4_entries(volatile tt_l1_ptr uint16_t* logits, uint32_t id0, volatile tt_l1_ptr uint32_t* top) {
    const volatile tt_l1_ptr uint32_t* w = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(logits);
    uint32_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (uint32_t i = 0; i < 16; ++i) {
        const uint32_t v = w[i];
        const uint32_t lo = v & 0xFFFFu, hi = v >> 16;
        const uint32_t klo = (lo & 0x8000u) ? (~lo & 0xFFFFu) : (lo | 0x8000u);
        const uint32_t khi = (hi & 0x8000u) ? (~hi & 0xFFFFu) : (hi | 0x8000u);
        top4_insert(0x80000000u | (klo << 8) | (255u - (id0 + 2 * i)), t0, t1, t2, t3);
        top4_insert(0x80000000u | (khi << 8) | (255u - (id0 + 2 * i + 1)), t0, t1, t2, t3);
    }
    top[0] = t0;
    top[1] = t1;
    top[2] = t2;
    top[3] = t3;
}

// The top-4 of n_lists x 4 merge entries and the Q16 softmax over their logits (as topk4_softmax_fast): 8 UINT16 ids
// then 8 BF16 scores (first 4 valid) to `stage`.
template <uint32_t n_lists>
inline void merge4_softmax(volatile tt_l1_ptr uint32_t* lists, volatile tt_l1_ptr uint16_t* stage) {
    uint32_t t[4] = {0, 0, 0, 0};
    for (uint32_t i = 0; i < 4 * n_lists; ++i) {
        top4_insert(lists[i], t[0], t[1], t[2], t[3]);
    }
    uint32_t ids[4];
    int32_t vals[4];
    for (uint32_t j = 0; j < 4; ++j) {
        ids[j] = 255u - (t[j] & 0xFFu);
        const uint32_t key = (t[j] >> 8) & 0xFFFFu;
        vals[j] = bf16_to_q16(static_cast<uint16_t>((key & 0x8000u) ? (key & 0x7FFFu) : (~key & 0xFFFFu)));
    }
    uint32_t e[4];
    uint32_t sum = 0;
    for (uint32_t j = 0; j < 4; ++j) {
        e[j] = exp_q16(vals[j] - vals[0]);
        sum += e[j];
    }
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage);
    st[0] = ids[0] | (ids[1] << 16);
    st[1] = ids[2] | (ids[3] << 16);
    st[2] = 0;
    st[3] = 0;
    const uint32_t d = sum >> 1;
    st[4] = q16_to_bf16((e[0] << 15) / d) | (static_cast<uint32_t>(q16_to_bf16((e[1] << 15) / d)) << 16);
    st[5] = q16_to_bf16((e[2] << 15) / d) | (static_cast<uint32_t>(q16_to_bf16((e[3] << 15) / d)) << 16);
    st[6] = 0;
    st[7] = 0;
}
}  // namespace router_topk

namespace router_topk {
// Multi-row router tail (DFlash verify, the fused MoE's router: dflash/moe_fused.py, MOE_EARLY_UNION), in parts so
// that BRISC and NCRISC split the rows and the union (all the gate|up / down cores need to start) goes out before the
// softmax weights. words = the [rows, 32] BF16 logits tile (row r: 16 values in face 0 at word 8 r, 16 in face 1 at
// word 8 (rows + r)). rows_ids: rows [r0, r1)'s sorted top-4 ids (ties to the lower id) and order keys (ids /
// keys[r * topk + j]); rows_softmax: their BF16 softmax weights from the keys (wq[r * topk + j]); union_ids: the
// union list uni[0] = U, uni[1 + i] = union id i (ascending); union_weights: the [rows, 32] weight matrix,
// wts[32 r + i] = row r's weight of union expert i (0 if not routed).
template <uint32_t topk, uint32_t rows>
inline void rows_ids(
    const uint32_t* words,
    uint32_t r0,
    uint32_t r1,
    volatile tt_l1_ptr uint32_t* ids,
    volatile tt_l1_ptr uint32_t* keys) {
    static_assert(topk == 4, "rows_ids: register top-4");
    for (uint32_t r = r0; r < r1; ++r) {
        uint32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, i0 = 0, i1 = 0, i2 = 0, i3 = 0;
        auto ins = [&](uint32_t key, uint32_t id) __attribute__((always_inline)) {
            if (key > b3) {
                if (key > b2) {
                    b3 = b2;
                    i3 = i2;
                    if (key > b1) {
                        b2 = b1;
                        i2 = i1;
                        if (key > b0) {
                            b1 = b0;
                            i1 = i0;
                            b0 = key;
                            i0 = id;
                        } else {
                            b1 = key;
                            i1 = id;
                        }
                    } else {
                        b2 = key;
                        i2 = id;
                    }
                } else {
                    b3 = key;
                    i3 = id;
                }
            }
        };
#pragma GCC unroll 2
        for (uint32_t f = 0; f < 2; ++f) {
            const uint32_t* w = words + 8 * (f * rows + r);
#pragma GCC unroll 8
            for (uint32_t q = 0; q < 8; ++q) {
                const uint32_t v = w[q];
                const uint32_t lo = v & 0xFFFF, hi = v >> 16;
                ins((lo & 0x8000) ? (~lo & 0xFFFF) : (lo | 0x8000), f * 16 + 2 * q);
                ins((hi & 0x8000) ? (~hi & 0xFFFF) : (hi | 0x8000), f * 16 + 2 * q + 1);
            }
        }
        ids[r * 4 + 0] = i0;
        ids[r * 4 + 1] = i1;
        ids[r * 4 + 2] = i2;
        ids[r * 4 + 3] = i3;
        keys[r * 4 + 0] = b0;
        keys[r * 4 + 1] = b1;
        keys[r * 4 + 2] = b2;
        keys[r * 4 + 3] = b3;
    }
}

template <uint32_t topk>
inline void rows_softmax(uint32_t r0, uint32_t r1, volatile tt_l1_ptr uint32_t* keys, volatile tt_l1_ptr uint32_t* wq) {
    for (uint32_t r = r0; r < r1; ++r) {
        int32_t v0 = 0;
        uint32_t e[topk];
        uint32_t sum = 0;
        for (uint32_t j = 0; j < topk; ++j) {
            const uint32_t k = keys[r * topk + j];
            const uint16_t b = (k & 0x8000) ? static_cast<uint16_t>(k & 0x7FFF) : static_cast<uint16_t>(~k & 0xFFFF);
            const int32_t v = bf16_to_q16(b);
            if (j == 0) {
                v0 = v;
            }
            e[j] = exp_q16(v - v0);
            sum += e[j];
        }
        for (uint32_t j = 0; j < topk; ++j) {
            wq[r * topk + j] = q16_to_bf16((e[j] << 15) / (sum >> 1));
        }
    }
}

// the union list ([U, ids ascending] UINT16) into uni_w; slot[id] = its union index
template <uint32_t topk, uint32_t rows>
inline uint32_t union_ids(volatile tt_l1_ptr uint32_t* ids, volatile tt_l1_ptr uint32_t* uni_w, uint32_t* slot) {
    uint32_t mask = 0;
    for (uint32_t i = 0; i < rows * topk; ++i) {
        mask |= 1u << ids[i];
    }
    uint16_t list[64];
    uint32_t u = 0;
    for (uint32_t m = mask; m; m &= m - 1) {
        const uint32_t e = __builtin_ctz(m);
        slot[e] = u;
        list[1 + u] = e;
        ++u;
    }
    list[0] = u;
    list[1 + u] = 0;
    const uint32_t uw = (u + 2) / 2;
    for (uint32_t i = 0; i < uw; ++i) {
        uni_w[i] = list[2 * i] | (static_cast<uint32_t>(list[2 * i + 1]) << 16);
    }
    return u;
}

template <uint32_t topk, uint32_t rows>
inline void union_weights(
    uint32_t u,
    const uint32_t* slot,
    volatile tt_l1_ptr uint32_t* ids,
    volatile tt_l1_ptr uint32_t* wq,
    volatile tt_l1_ptr uint32_t* wts_w) {
    const uint32_t ww = (u + 1) / 2;
    for (uint32_t r = 0; r < rows; ++r) {
        uint16_t wt[34];
        for (uint32_t i = 0; i < 2 * ww; ++i) {
            wt[i] = 0;
        }
        for (uint32_t j = 0; j < topk; ++j) {
            wt[slot[ids[r * topk + j]]] = static_cast<uint16_t>(wq[r * topk + j]);
        }
        for (uint32_t i = 0; i < ww; ++i) {
            wts_w[16 * r + i] = wt[2 * i] | (static_cast<uint32_t>(wt[2 * i + 1]) << 16);
        }
    }
}
}  // namespace router_topk
