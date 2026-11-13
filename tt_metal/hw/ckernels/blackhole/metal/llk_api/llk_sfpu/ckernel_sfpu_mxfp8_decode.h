// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// SFPU decoders for MXFP8 (E4M3 or E5M2) stored as remapped bfloat8_b, over one tile in Dest.
//
// The unpacker normalizes the bfloat8_b 7-bit magnitude m: the FP32 image is 2^(q+k) * (1 + w/64), w = mantissa
// bits 22..17. The decode keeps the top P bits of w (the MXFP8 mantissa: P = 3 for E4M3, 2 for E5M2), drops the
// remaining bits b (bits 19..17 or 20..17) and adds g(b) = 2b - popcount(b) to the exponent:
//     out = U - (b << 17) + (g(b) << 23)
// g depends on b only, so the decode works for any shared exponent (the MXFP8 scale lives in it). The host
// stores magnitude code c as m = ENC[c], the inverse of this decode on the integers 0..127.
//
// tt_metal/tt-llk/tests/sources/mxfp8_bf8_decode_test.cpp checks both formats bit-exact against MXFP8 for every
// e8m0 scale.

#pragma once

#include "ckernel.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Decode the whole tile at the current Dest address (after _llk_math_eltwise_sfpu_start_), 32 vectors of 32 lanes.
// E4M3: b = bits 19..17 (3 bits), g(b) in 0..11 read from a nibble table with a register shift. Per vector: load,
// shft (s = U >> 15: low 5 bits = 4b), shft by register (LUT << s), shft (>> 28), shft (<< 23), and (clear b),
// iadd, store.
// E5M2: b = bits 20..17 (4 bits). out = U + 127 * t - (popcount(b) << 23), t = U & (15 << 17), popcount from a
// 2-bit table (popcount(15) wraps to 0, but b = 15 only occurs on the inf/NaN slots the host never writes).
template <bool E5M2>
inline void mxfp8_decode_tile() {
    if constexpr (!E5M2) {
        const sfpi::vUInt lut = 0x013478ABu;   // nibble b at bits 31-4b..28-4b: g = 0, 1, 3, 4, 7, 8, 10, 11
        const sfpi::vUInt keep = 0xFFF1FFFFu;  // ~(7 << 17)
#pragma GCC unroll 8
        for (int d = 0; d < 32; d++) {
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
            sfpi::vUInt s = u >> 15;
            sfpi::vUInt g = lut << s;
            g = (g >> 28) << 23;
            u = (u & keep) + g;
            sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
            sfpi::dst_reg++;
        }
    } else {
        const sfpi::vUInt lut = 0x166B6BBCu;  // 2-bit entry b at bits 31-2b..30-2b: popcount(b) & 3
        const sfpi::vUInt mb = 0x001E0000u;
#pragma GCC unroll 8
        for (int d = 0; d < 32; d++) {
            sfpi::vUInt u = sfpi::as<sfpi::vUInt>(sfpi::vFloat(sfpi::dst_reg[0]));
            sfpi::vUInt t = u & mb;
            sfpi::vUInt s = u >> 16;
            sfpi::vUInt p = lut << s;
            p = (p >> 30) << 23;
            u = u + (t << 7) - t - p;
            sfpi::dst_reg[0] = sfpi::as<sfpi::vFloat>(u);
            sfpi::dst_reg++;
        }
    }
}

}  // namespace ckernel::sfpu
