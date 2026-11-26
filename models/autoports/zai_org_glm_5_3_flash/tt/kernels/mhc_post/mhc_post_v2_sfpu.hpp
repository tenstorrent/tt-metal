// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stage 14c mhc_post v2 (MPOST_V2, the per-tile path, t > 4): the post-mix of one column tile in ONE fused SFPI pass,
// in place over the 4 residual tiles in DST, only over the 4-row bands that hold tokens (t < T; the other bands of
// the outputs are written zero, as v1 computes 0 there from the zero coefficient rows):
//   out_s = y * post_s + sum_i x_i * C[i][s]        (per token = per tile row; fp32, FMA rounding)
// REQUIRES dst_full_sync_en (8 fp32 DST tiles; half sync gives 4): only the MkLayer / MkSparseLayer owner composites
// build it (mhc_post.py parts(fused_v2=True)).
// DST: slot 0 = y (+ y2), slots 1..4 = x_0..3 -> out_0..3, slot 5 = the compact coefficient tile (plane k = coefficient
// k: post 0..3 | C[i][s] at 4 + 4 i + s; lane (i, c) = token 4 c + i, written by the v2 reader), slot 6 = the 20
// per-band coefficient broadcasts (planes 0..19) and the column masks (planes 24..31). Plane p of a tile =
// (2 fr + fc) * 8 + 2 band + par: rows 16 fr + 4 band + i = tokens 4 bb + i of band bb = 4 fr + band (the kda_rec v2
// plane map); the coefficient of the band's 4 tokens sits in SFPU column bb of the coefficient plane, one per sub-row,
// and is spread over the sub-row (masked rotate-and-add) -> no per-row column broadcast tile ops.
#pragma once

#ifdef TRISC_MATH
#include "sfpi.h"

namespace ckernel::sfpu {

namespace mpv2 {
constexpr int PT = 32;
constexpr int Y = 0, X = 1 * PT, COEF = 5 * PT, BC = 6 * PT, MASK = 6 * PT + 24;
constexpr int sp(int bb, int cg) { return (2 * (bb >> 2) + (cg >> 1)) * 8 + 2 * (bb & 3) + (cg & 1); }
}  // namespace mpv2

sfpi_inline sfpi::vFloat mpv2_ror(sfpi::vFloat v) { return sfpi::vFloat(sfpi::subvec_shflror1(v)); }

// every lane of a sub-row <- the sum of its 8 lanes (spreads the single non-zero lane of a masked value)
sfpi_inline sfpi::vFloat mpv2_rowsum8(sfpi::vFloat v) {
    v = v + mpv2_ror(v);
    v = v + mpv2_ror(mpv2_ror(v));
    const sfpi::vFloat t = mpv2_ror(mpv2_ror(v));
    v = v + mpv2_ror(mpv2_ror(t));
    return v;
}

// nb = the bands holding tokens (ceil(T / 4))
inline void mhc_post_v2_mix(const uint32_t nb) {
    using namespace mpv2;
    const sfpi::vInt col = (sfpi::vConstTileId >> 1) & 7;
#pragma GCC unroll 8
    for (int c = 0; c < 8; c++) {
        sfpi::vFloat m = 0.0f;
        v_if(col == c) { m = 1.0f; }
        v_endif;
        sfpi::dst_reg[MASK + c] = m;
    }
#pragma GCC unroll 8
    for (int bb = 0; bb < 8; bb++) {
        if (static_cast<uint32_t>(bb) < nb) {
            const sfpi::vFloat mk = sfpi::dst_reg[MASK + bb];
#pragma GCC unroll 20
            for (int k = 0; k < 20; k++) {
                sfpi::dst_reg[BC + k] = mpv2_rowsum8(sfpi::dst_reg[COEF + k] * mk);
            }
#pragma GCC unroll 4
            for (int cg = 0; cg < 4; cg++) {
                const int p = sp(bb, cg);
                const sfpi::vFloat yv = sfpi::dst_reg[Y + p];
                const sfpi::vFloat x0 = sfpi::dst_reg[X + 0 * PT + p], x1 = sfpi::dst_reg[X + 1 * PT + p];
                const sfpi::vFloat x2 = sfpi::dst_reg[X + 2 * PT + p], x3 = sfpi::dst_reg[X + 3 * PT + p];
#pragma GCC unroll 4
                for (int s = 0; s < 4; s++) {
                    sfpi::vFloat acc = yv * sfpi::dst_reg[BC + s];
                    acc = acc + x0 * sfpi::dst_reg[BC + 4 + s];
                    acc = acc + x1 * sfpi::dst_reg[BC + 8 + s];
                    acc = acc + x2 * sfpi::dst_reg[BC + 12 + s];
                    acc = acc + x3 * sfpi::dst_reg[BC + 16 + s];
                    sfpi::dst_reg[X + s * PT + p] = acc;
                }
            }
        } else {
#pragma GCC unroll 4
            for (int cg = 0; cg < 4; cg++) {
#pragma GCC unroll 4
                for (int s = 0; s < 4; s++) {
                    sfpi::dst_reg[X + s * PT + sp(bb, cg)] = 0.0f;
                }
            }
        }
    }
}

// Stage 14b P1-3 option 2 (MPOST_P_FMA, the PACKED path, t <= 4; compute_p.cpp): OUT = A * B + C over whole tiles as
// ONE SFPMAD per element. mhc_post_v2_mix compiles (sfpi 7.86) to: x_0 * C[0][s] as SFPMUL, then y * post_s + that as
// ONE SFPMAD, then + x_i * C[i][s] as one SFPMAD each (i = 1..3); compute_p.cpp replays that order with this, so a
// narrow width's rows are bitwise a native width's under pv2. DST tile indices are template arguments (SFPLOAD
// immediates); vFloat locals (dst_reg * dst_reg has no operator).
template <int OUT, int A, int B, int C>
inline void mhc_post_p_mad() {
    constexpr int PT = 32;
#pragma GCC unroll 8
    for (int p = 0; p < PT; p++) {
        const sfpi::vFloat a = sfpi::dst_reg[A * PT + p];
        const sfpi::vFloat b = sfpi::dst_reg[B * PT + p];
        const sfpi::vFloat c = sfpi::dst_reg[C * PT + p];
        sfpi::dst_reg[OUT * PT + p] = a * b + c;
    }
}

// the same with the column broadcast fused in: OUT = A * bcast_col(COEF) (+ C when MAD) per element, COEF = a packed
// coefficient tile (column 0 = the row's coefficient, every other column +0: reader_p.cpp zeroes the CB). Per 4-row
// band bb, the column-0 plane's sub-row sum (mpv2_rowsum8: the coefficient + seven +0 = exact) is the broadcast for
// the band's 4 column groups -- no fill / LLK broadcast pass per term (B=1 cost).
template <int OUT, int A, int COEF, int C, bool MAD>
inline void mhc_post_p_bcast_mad() {
    using namespace mpv2;
#pragma GCC unroll 8
    for (int bb = 0; bb < 8; bb++) {
        const sfpi::vFloat bc = mpv2_rowsum8(sfpi::dst_reg[COEF * PT + sp(bb, 0)]);
#pragma GCC unroll 4
        for (int cg = 0; cg < 4; cg++) {
            const int p = sp(bb, cg);
            const sfpi::vFloat a = sfpi::dst_reg[A * PT + p];
            if constexpr (MAD) {
                const sfpi::vFloat c = sfpi::dst_reg[C * PT + p];
                sfpi::dst_reg[OUT * PT + p] = a * bc + c;
            } else {
                sfpi::dst_reg[OUT * PT + p] = a * bc;
            }
        }
    }
}

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
