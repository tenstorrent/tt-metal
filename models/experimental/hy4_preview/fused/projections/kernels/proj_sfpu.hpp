// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// SFPU finishing of a projection unit (proj_compute.cpp), math thread only: everything after the matmul stays FP32
// in DEST and in the SFPU's registers.
//
// DEST (full sync, FP32: 512 rows). Addresses are sfpi::dst_reg indices ("units", 2 DEST rows each). One SFPU
// register is 4 DEST rows x 8 lanes: unit 2 k is rows 4 k .. 4 k + 3 on the even columns, unit 2 k + 1 the same
// rows on the odd columns (measured on the board by fused/ihc_site, tests/test_ihc_site.py test_probe_lanes).
//   output tile t (custom_mm Dense layout: 32 rows apart, face 1 at row 16): unit 16 t + 8 h + 2 g + p is token rows
//       4 g .. 4 g + 3, face h (columns 16 h ..), column parity p
//   a tile copied to standard slot j (64 rows apart): unit 32 j + ...
//       7   the constant page (FP32 [32, 32], written by the unit's data-movement kernel): face 0 row r = scale[r] on
//           all columns, row 8 + r = the 8 lane sums of token row r on the even columns
//       6   FIN_STAT: the statistic tile going out (packed at the standard tile stride: a pack format change
//           restores it)
//       5, 6  cos, sin as [ROWS, 32] tiles (layout.py rope_tile): unit 32 j + 8 t + 2 g + h holds the 8 pairs of
//           (rope tile t, face h)
//       2 ..  the gate tiles of FIN_GATE, tile i in slot 2 + i
// An interleaved RoPE pair (2 i, 2 i + 1) is lane i of the even register and lane i of the odd register of the
// same rows, so the rotation is elementwise between the two.

#pragma once

#ifdef TRISC_MATH

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_sqrt.h"
#include "proj_common.hpp"
#include "sfpi.h"

namespace hy4proj {

constexpr int U_CONST = 7 * 32;
#if ROWS == 8
constexpr int U_LANES = U_CONST + 4;  // rows 8 .. 15 of the constant page's face 0
#else
// 16 token rows (hifi_mm16): an output tile fills the 32 rows of the slot the 8-row form already gives it, so every
// address below holds with the row group g running to 4; the constant page's face 0 is all scales (rows 0 .. 15)
// and its lane sums sit in face 2 (page rows 16 + r).
constexpr int U_LANES = U_CONST + 16;
#endif
constexpr int U_NORM = U_CONST + 8;  // scratch (face 1 of the constant page): the row multipliers of FIN_NORM
constexpr int U_COS = 5 * 32;
constexpr int U_SIN = 6 * 32;
constexpr int U_GATE = 2 * 32;
constexpr int U_STAT = 6 * 32;

// Sum of the 8 lanes of every row, left on all 8 lanes (derived from fused/ihc_site/kernels/ihc_sfpu.hpp hsum8).
sfpi_inline sfpi::vFloat hsum8(sfpi::vFloat v) {
    sfpi::vFloat r = sfpi::subvec_shflror1(v);
    v = v + r;
    r = sfpi::subvec_shflror1(v);
    r = sfpi::subvec_shflror1(r);
    v = v + r;
    r = sfpi::subvec_shflror1(v);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    r = sfpi::subvec_shflror1(r);
    v = v + r;
    return v;
}

// To nearest at 16 (BF16) or 17 (7 significant bits) dropped mantissa bits; halves away from zero.
template <int RND>
sfpi_inline sfpi::vFloat round_to(sfpi::vFloat v) {
    if constexpr (RND == static_cast<int>(RND_NONE)) {
        return v;
    } else {
        constexpr int drop = RND == static_cast<int>(RND_BF16) ? 16 : 17;
        sfpi::vUInt u = sfpi::as<sfpi::vUInt>(v);
        u = u + (1 << (drop - 1));
        u = u >> drop;
        u = u << drop;
        return sfpi::as<sfpi::vFloat>(u);
    }
}

// The unit's CT output tiles in place. FIN, RND: proj_common.hpp. PARAM: the width the lane sums are a mean over
// (FIN_NORM, FIN_NORM_ROPE), or the FP32 bits of the constant (FIN_CONST).
template <int CT, int FIN, int RND, uint32_t PARAM>
inline void finish() {
    constexpr bool norm = FIN == static_cast<int>(FIN_NORM) || FIN == static_cast<int>(FIN_NORM_ROPE);
    constexpr bool rope = FIN == static_cast<int>(FIN_NORM_ROPE) || FIN == static_cast<int>(FIN_SCALE_ROPE);
    constexpr bool scaled = fin_has_const(FIN);
    constexpr int U_MUL = norm ? U_NORM : U_CONST;
    constexpr int NORM_N = norm ? static_cast<int>(PARAM) : 1;

    if constexpr (norm) {
#if ROWS == 8
#pragma GCC unroll 2
        for (int g = 0; g < 2; g++) {
#else
#pragma GCC unroll 4
        for (int g = 0; g < 4; g++) {
#endif
            sfpi::vFloat a = sfpi::dst_reg[U_LANES + 2 * g];
            a = hsum8(a);
            a = a * (1.0f / NORM_N) + 1e-6f;
            sfpi::vFloat rs = ckernel::sfpu::_calculate_sqrt_body_<false, true, false>(a);
            sfpi::vFloat s = sfpi::dst_reg[U_CONST + 2 * g];
            sfpi::dst_reg[U_NORM + 2 * g] = rs * s;
        }
    }

#if ROWS == 8
#pragma GCC unroll 2
    for (int g = 0; g < 2; g++) {
#else
#pragma GCC unroll 4
    for (int g = 0; g < 4; g++) {
#endif
        if constexpr (FIN == static_cast<int>(FIN_STAT)) {
            sfpi::vFloat acc = 0.0f;
#pragma GCC unroll 16
            for (int t = 0; t < CT; t++) {
#pragma GCC unroll 4
                for (int hp = 0; hp < 4; hp++) {
                    const int u = 16 * t + 8 * (hp >> 1) + 2 * g + (hp & 1);
                    sfpi::vFloat v = sfpi::dst_reg[u];
                    sfpi::vFloat s = sfpi::dst_reg[U_MUL + 2 * g];
                    v = round_to<RND>(v * s);
                    sfpi::dst_reg[u] = v;
                    acc = v * v + acc;
                }
            }
            // the statistic tile: the lane sums on the even columns of face 0, zero elsewhere
            sfpi::vFloat z = 0.0f;
            sfpi::dst_reg[U_STAT + 2 * g] = acc;
            sfpi::dst_reg[U_STAT + 2 * g + 1] = z;
            sfpi::dst_reg[U_STAT + 8 + 2 * g] = z;
            sfpi::dst_reg[U_STAT + 8 + 2 * g + 1] = z;
        } else if constexpr (rope) {
#pragma GCC unroll 16
            for (int t = 0; t < CT; t++) {
#pragma GCC unroll 2
                for (int h = 0; h < 2; h++) {
                    const int u = 16 * t + 8 * h + 2 * g;
                    sfpi::vFloat e = sfpi::dst_reg[u];
                    sfpi::vFloat o = sfpi::dst_reg[u + 1];
                    sfpi::vFloat c = sfpi::dst_reg[U_COS + 8 * t + 2 * g + h];
                    sfpi::vFloat sn = sfpi::dst_reg[U_SIN + 8 * t + 2 * g + h];
                    // (e, o) -> (e cos - o sin, o cos + e sin)
                    sfpi::vFloat ne = e * c;
                    sfpi::vFloat tmp = o * sn;
                    ne = ne - tmp;
                    tmp = e * sn;
                    o = o * c + tmp;
                    sfpi::vFloat s = sfpi::dst_reg[U_MUL + 2 * g];
                    sfpi::dst_reg[u] = round_to<RND>(ne * s);
                    sfpi::dst_reg[u + 1] = round_to<RND>(o * s);
                }
            }
        } else {
#pragma GCC unroll 16
            for (int t = 0; t < CT; t++) {
#pragma GCC unroll 4
                for (int hp = 0; hp < 4; hp++) {
                    const int u = 16 * t + 8 * (hp >> 1) + 2 * g + (hp & 1);
                    sfpi::vFloat v = sfpi::dst_reg[u];
                    if constexpr (scaled) {
                        sfpi::vFloat s = sfpi::dst_reg[U_MUL + 2 * g];
                        v = v * s;
                    }
                    if constexpr (FIN == static_cast<int>(FIN_SIGMOID)) {
                        v = ckernel::sfpu::_sfpu_sigmoid_<true>(v);
                    }
                    if constexpr (FIN == static_cast<int>(FIN_CONST)) {
                        sfpi::vFloat k = sfpi::as<sfpi::vFloat>(sfpi::vUInt(PARAM));
                        v = v * k;
                    }
                    if constexpr (FIN == static_cast<int>(FIN_GATE)) {
                        sfpi::vFloat gt = sfpi::dst_reg[U_GATE + 32 * t + 8 * (hp >> 1) + 2 * g + (hp & 1)];
                        v = v * gt;
                    }
                    sfpi::dst_reg[u] = round_to<RND>(v);
                }
            }
        }
    }
}

}  // namespace hy4proj

#endif  // TRISC_MATH
