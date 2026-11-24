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

// row `row` of the packed tile at t from row src_row of the bf16 tile at s
FORCE_INLINE void patch_bfp4_row(uint32_t t, uint32_t row, uint32_t s, uint32_t src_row, bool rnd = true) {
    volatile tt_l1_ptr uint8_t* ex = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(t);
    volatile tt_l1_ptr uint32_t* man = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t + 64);
    volatile tt_l1_ptr uint16_t* src = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(s);
    for (uint32_t h = 0; h < 2; h++) {
        const uint32_t f = (row < 16 ? 0 : 2) + h, r = row & 15;
        const uint32_t sf = (src_row < 16 ? 0 : 2) + h, sr = src_row & 15;
        const uint32_t emax = ex[f * 16 + r];
        volatile tt_l1_ptr uint16_t* b = src + sf * 256 + sr * 16;
        uint32_t w0 = 0, w1 = 0;
        for (uint32_t j = 0; j < 8; j++) {
            w0 |= rtn_bfp4_nibble(b[j], emax, rnd) << (4 * j);
            w1 |= rtn_bfp4_nibble(b[8 + j], emax, rnd) << (4 * j);
        }
        man[f * 32 + r * 2] = w0;
        man[f * 32 + r * 2 + 1] = w1;
    }
}
