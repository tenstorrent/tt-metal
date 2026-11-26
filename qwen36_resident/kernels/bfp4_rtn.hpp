// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// S0-8: re-encode a row of a PACKED bfp4_b tile to nearest from its bf16 source (the device packer truncates toward
// zero). Assumed tile layout (the host packer's, blockfloat_common.cpp): 64 exponent bytes (face f row r at f * 16 +
// r), then the faces' 16 x 16 values as 4-bit sign + magnitude nibbles (128 B a face, 8 B a face row, element j at
// nibble j of its word, LSB first); a value is m * 2^(e - 129). The shared exponents stay (rounding with m capped at 7
// keeps the row's max exponent). tests/bfp4_patch_probe.py checks it on the device.
#pragma once

#include <cstdint>

FORCE_INLINE uint32_t rtn_bfp4_nibble(uint32_t b, uint32_t emax, bool rnd = true) {
    // (a zero magnitude is stored as +0: the device unpacker decodes the nibble 0x8 — sign, magnitude 0, which its
    // packer never writes — as -inf, which the next re-pack of the tail tile spreads over the whole face row)
    const uint32_t e = (b >> 7) & 0xFF, sign = (b >> 12) & 0x8;
    if (e == 0 || e > emax || emax < 3) {
        return 0;
    }
    const uint32_t sh = emax - e + 5;
    if (sh >= 16) {
        return 0;
    }
    const uint32_t m = (128 + (b & 0x7F) + (rnd ? (1u << (sh - 1)) : 0u)) >> sh;
    return m == 0 ? 0 : sign | (m > 7 ? 7 : m);
}

// row `row` of the packed tile at t from row src_row of the bf16 tile at s. The face row's 8 source words are loaded
// once (two bf16 a word) and each value takes the exponent distance d = emax - e: d 0 .. 3 give m (128 + mantissa +
// 2^(d + 4)) >> (d + 5) in 4..7 (capped), 2..4, 1..2, 1 — never 0 — and d > 3 (or a zero / denormal) gives 0 (the
// same values as rtn_bfp4_nibble; the 32-row step's tail writer re-encodes a K and a V row each user, S0-9)
FORCE_INLINE uint32_t rtn_bfp4_fast(uint32_t b, uint32_t emax, bool rnd) {
    const uint32_t e = (b >> 7) & 0xFF, d = emax - e;
    if (e == 0 || d > 3) {
        return 0;
    }
    const uint32_t sh = d + 5;
    const uint32_t m = (128 + (b & 0x7F) + (rnd ? (1u << (sh - 1)) : 0u)) >> sh;
    return m == 0 ? 0 : ((b >> 12) & 0x8) | (m > 7 ? 7 : m);  // (m 0 only truncating, kv_k_patch 2: +0, never 0x8)
}
FORCE_INLINE void patch_bfp4_row(uint32_t t, uint32_t row, uint32_t s, uint32_t src_row, bool rnd = true) {
    volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(t);
    volatile tt_l1_ptr uint32_t* man = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + 64);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s);
    for (uint32_t h = 0; h < 2; h++) {
        const uint32_t f = (row < 16 ? 0 : 2) + h, r = row & 15;
        const uint32_t sf = (src_row < 16 ? 0 : 2) + h, sr = src_row & 15;
        const uint32_t emax = ex[f * 16 + r];
        uint32_t w[8];
        for (uint32_t i = 0; i < 8; i++) {
            w[i] = src[sf * 128 + sr * 8 + i];
        }
        uint32_t w0 = 0, w1 = 0;
        if (emax >= 3) {
            for (uint32_t i = 0; i < 4; i++) {
                w0 |= (rtn_bfp4_fast(w[i] & 0xFFFF, emax, rnd) | (rtn_bfp4_fast(w[i] >> 16, emax, rnd) << 4))
                      << (8 * i);
                w1 |= (rtn_bfp4_fast(w[4 + i] & 0xFFFF, emax, rnd) | (rtn_bfp4_fast(w[4 + i] >> 16, emax, rnd) << 4))
                      << (8 * i);
            }
        }
        man[f * 32 + r * 2] = w0;
        man[f * 32 + r * 2 + 1] = w1;
    }
}
