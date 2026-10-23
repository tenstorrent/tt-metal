// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Fixed-point helpers of the decode sampling head (tt/device_sampler.py). The data movement cores have no FPU: logits
// are Q24, a probability relative to a larger logit, 2^((logit - larger) * log2(e) / T), is Q30 (absolute error
// ~1e-9), and sums of those are uint64.

#pragma once

#include <cstdint>

namespace sample {

constexpr uint32_t kOne30 = 1u << 30;

// 2^(j / 64) for j = 0..63, Q30.
constexpr uint32_t kExp2Table[64] = {
    1073741824, 1085434106, 1097253708, 1109202018, 1121280436, 1133490379, 1145833280, 1158310587,
    1170923762, 1183674286, 1196563654, 1209593378, 1222764986, 1236080024, 1249540052, 1263146652,
    1276901417, 1290805962, 1304861917, 1319070932, 1333434672, 1347954824, 1362633090, 1377471191,
    1392470869, 1407633882, 1422962010, 1438457051, 1454120821, 1469955159, 1485961921, 1502142985,
    1518500250, 1535035634, 1551751076, 1568648537, 1585730000, 1602997467, 1620452965, 1638098541,
    1655936265, 1673968228, 1692196547, 1710623359, 1729250827, 1748081133, 1767116489, 1786359126,
    1805811301, 1825475297, 1845353420, 1865448001, 1885761398, 1906295993, 1927054196, 1948038440,
    1969251188, 1990694927, 2012372174, 2034285470, 2056437387, 2078830522, 2101467502, 2124350982};
constexpr uint64_t kLn2Q30 = 744261118;  // ln(2) in Q30

inline bool better(uint32_t key, uint32_t id, uint32_t other_key, uint32_t other_id) {
    return key > other_key || (key == other_key && id < other_id);
}

// fp32 bits of a sort key (inverse of sample_scan.cpp's sort_key).
inline uint32_t key_bits(uint32_t key) { return (key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key; }

// An fp32 (bits) times 2^frac, toward zero, as int64; magnitudes past 2^62 saturate.
inline int64_t to_fixed(uint32_t bits, int frac) {
    const int e = static_cast<int>((bits >> 23) & 0xFF);
    if (e == 0) {
        return 0;
    }
    const int64_t m = static_cast<int64_t>((bits & 0x7FFFFF) | 0x800000);
    const int sh = e - 150 + frac;
    int64_t v;
    if (sh >= 0) {
        v = sh >= 39 ? (int64_t{1} << 62) : (m << sh);
    } else {
        v = sh <= -32 ? 0 : (m >> -sh);
    }
    return (bits & 0x80000000u) ? -v : v;
}

// 2^t for t <= 0 given in Q24, as Q30 (0 below 2^-31).
inline uint64_t exp2_q30(int64_t t) {
    const int64_t n = t >> 24;  // floor
    if (n < -31) {
        return 0;
    }
    const uint32_t f = static_cast<uint32_t>(t & 0xFFFFFF);  // [0, 1) in Q24
    const uint32_t j = f >> 18;                              // 2^(j / 64) from the table
    const uint64_t g = f & 0x3FFFF;                          // the rest, < 1/64, Q24
    // e^h with h = g * ln2 < 0.011: 1 + h + h^2 / 2 + h^3 / 6 (error < 1e-9), Q30.
    const uint64_t h = (g * kLn2Q30) >> 24;
    const uint64_t h2 = (h * h) >> 30;
    const uint64_t h3 = (h2 * h) >> 30;
    const uint64_t eh = kOne30 + h + (h2 >> 1) + static_cast<uint32_t>(h3) / 6u;
    const uint64_t p = (static_cast<uint64_t>(kExp2Table[j]) * eh) >> 30;  // 2^f in Q30, [2^30, 2^31)
    return p >> static_cast<uint32_t>(-n);
}

// a * b / 2^24 for b <= 2^24 (a Q24 fraction) without overflow for a < 2^63.
inline uint64_t mul_q24(uint64_t a, uint32_t b) { return (a >> 24) * b + (((a & 0xFFFFFF) * b) >> 24); }

// a * b / 2^30 for b <= 2^30 (a Q30 fraction) without overflow for a < 2^63.
inline uint64_t mul_q30(uint64_t a, uint64_t b) { return (a >> 30) * b + (((a & (kOne30 - 1)) * b) >> 30); }

// Relative probabilities 2^((v - top) * log2(e) / T) of fp32 logits v <= top, for one temperature.
struct Rel {
    uint64_t scale_m;  // log2(e) / T = scale_m * 2^scale_e, scale_m the 24-bit mantissa
    int scale_e;
    int64_t top;  // Q24

    Rel(uint32_t scale_bits, uint32_t top_bits) :
        scale_m((scale_bits & 0x7FFFFF) | 0x800000),
        scale_e(static_cast<int>((scale_bits >> 23) & 0xFF) - 150),
        top(to_fixed(top_bits, 24)) {}

    // Q30; v given as fp32 bits.
    uint64_t operator()(uint32_t v_bits) const {
        const int64_t d = to_fixed(v_bits, 24) - top;  // <= 0, Q24
        if (d <= -(int64_t{1} << 31)) {                // 128 below: Q30 zero unless T > 4
            return d < -(int64_t{1} << 34) ? 0 : exp2_q30(((d >> 8) * static_cast<int64_t>(scale_m)) >> (-scale_e - 8));
        }
        const int64_t prod = static_cast<int64_t>(static_cast<int32_t>(d)) * static_cast<int32_t>(scale_m);
        return exp2_q30(scale_e >= 0 ? (prod << scale_e) : (prod >> -scale_e));
    }
};

// fp32 bits of a value in [0, 2^33) as Q30, toward zero.
inline uint64_t fp32_to_q30(uint32_t bits) {
    const int64_t v = to_fixed(bits, 30);
    return v < 0 ? 0 : static_cast<uint64_t>(v);
}

// The next uniform in [0, 1) of a splitmix64 stream, Q24 (its output's top 24 bits); state: 64 bits, low word first.
inline uint32_t next_uniform_q24(uint32_t* state) {
    uint64_t x = (static_cast<uint64_t>(state[1]) << 32 | state[0]) + 0x9E3779B97F4A7C15ull;
    state[0] = static_cast<uint32_t>(x);
    state[1] = static_cast<uint32_t>(x >> 32);
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    x ^= x >> 31;
    return static_cast<uint32_t>(x >> 40);
}

}  // namespace sample
