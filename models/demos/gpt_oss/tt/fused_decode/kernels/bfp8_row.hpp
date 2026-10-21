// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// BF16 -> BFP8_b for one 16-value face row (the multi-row QKV's K / V cache rows, kernels/stream_linear_writer.cpp;
// KV_SPLIT: half of the rows on the reader, kernels/stream_linear_reader.cpp).

#pragma once

#include <stdint.h>

namespace bfp8_row {
// One BF16 value (bits b) as a BFP8_b byte under shared exponent `shared`: sign | 7-bit magnitude = the 8-bit
// significand (hidden bit included) shifted right by shared - exponent + 1, rounded to nearest even and saturated (the
// rounding of tt_metal's host BFP8 packer); zero / denormal inputs and magnitudes that round to 0 give 0.
inline uint32_t byte(uint32_t b, uint32_t shared) {
    // branchless: round to nearest even as (sig + half - 1 + lsb) >> shift (exhaustively equal to the branchy form)
    const uint32_t e = (b >> 7) & 0xFF;
    const uint32_t shift_ = shared - e + 1;
    const uint32_t valid = (e != 0) & (shift_ <= 8);
    const uint32_t shift = valid ? shift_ : 8;
    const uint32_t sig = 0x80 | (b & 0x7F);
    uint32_t q = (sig + (1u << (shift - 1)) - 1 + ((sig >> shift) & 1)) >> shift;
    q -= q >> 7;  // 128 -> 127 (saturate)
    q = valid ? q : 0;
    return q ? (((b >> 15) << 7) | q) : 0;
}

// 16 BF16 values (8 words at in) -> the shared exponent and 4 mantissa words
inline void half_row(volatile tt_l1_ptr uint32_t* in, uint32_t& shared, uint32_t* man) {
    uint32_t v[16];
    shared = 0;
    for (uint32_t i = 0; i < 8; ++i) {
        const uint32_t w = in[i];
        v[2 * i] = w & 0xFFFF;
        v[2 * i + 1] = w >> 16;
        const uint32_t e0 = (w >> 7) & 0xFF;
        const uint32_t e1 = (w >> 23) & 0xFF;
        shared = e0 > shared ? e0 : shared;
        shared = e1 > shared ? e1 : shared;
    }
    for (uint32_t q = 0; q < 4; ++q) {
        man[q] = byte(v[4 * q], shared) | (byte(v[4 * q + 1], shared) << 8) | (byte(v[4 * q + 2], shared) << 16) |
                 (byte(v[4 * q + 3], shared) << 24);
    }
}

// the shared exponent + mantissas of half hh of row rr into the BFP8_b 32x32 tile at tile (64 exponent bytes
// face-major then row, then 4 x 256 mantissa bytes)
inline void put(uint32_t tile, uint32_t rr, uint32_t hh, uint32_t shared, const uint32_t* man) {
    const uint32_t face = (rr >> 4) * 2 + hh;
    const uint32_t frow = rr & 15;
    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(tile)[face * 16 + frow] = shared;
    volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile + 64 + face * 256 + frow * 16);
    for (uint32_t q = 0; q < 4; ++q) {
        m[q] = man[q];
    }
}
}  // namespace bfp8_row
