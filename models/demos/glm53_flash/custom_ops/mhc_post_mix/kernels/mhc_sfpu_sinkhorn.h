// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dest-resident Sinkhorn for the fused pre-block (TRISC_MATH, fp32 dest; geometry and register conventions of
// mhc_sfpu_x8.h, included first). The comb logits' exp sits in dest tile `m`: coefficient i*n+j of a token row in
// column i*n+j of the LEFT face (n = 4). Within a 4-row band one slot load holds the even columns (lane c = column
// 2c), the next (+ODD) the odd ones, so for the 4x4 matrix of a row
//   row sum over j of row i      = lanes {2i, 2i+1} of the even slot + the same lanes of the odd slot,
//   column sum over i of column j = lanes j/2 + 2i (stride 2, one parity class) of ONE slot,
// both subvector rotate-adds (SFPSHFT2 ROR1 moves lane c to lane c+1, cyclic over the 8 lanes), no matmul, no
// pack. Reciprocals: SFPARECIP + two Newton steps y <- y (2 - x y). Only the bands of the live rows are touched
// (pad rows are ignored downstream); the right face is never read or written.
// Register use: L0 even data, L1 odd data, L2 sums / reciprocal, L3 and L5 scratch, L4 the odd-lane mask (1.0 on
// lanes 1, 3, 5, 7), L6 -1.0, L7 eps. Dependent instructions sit >= 2 issue slots apart (SFPNOP).
#pragma once
#include "mhc_sfpu_x8.h"

#ifdef TRISC_MATH
namespace mhc_sfpu {

// dst <- src rotated right by k lanes within every 8-lane subvector (lane c -> lane c + k)
FORCE_INLINE void sk_ror(uint32_t src, uint32_t dst, uint32_t k) {
    TT_SFPSHFT2(0, src, dst, ROR1);
    TTI_SFPNOP;
    for (uint32_t i = 1; i < k; ++i) {
        TT_SFPSHFT2(0, dst, dst, ROR1);
        TTI_SFPNOP;
    }
}

// L2 <- 1 / L2 (SFPARECIP, then two Newton steps y <- y + y (1 - x y)); clobbers L3, L5; L6 = -1.0
FORCE_INLINE void sk_recip_l2() {
    TTI_SFPARECIP(0, L2, L5, sfpi::SFPARECIP_MOD1_RECIP);  // y
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPMAD(L2, L5, L6, L3, 0);  // t = x y - 1
    TTI_SFPNOP;
    TTI_SFPMUL(L3, L6, C0, L3, 0);  // -t
    TTI_SFPNOP;
    TTI_SFPMAD(L5, L3, L5, L5, 0);  // y <- y (-t) + y = y (2 - x y)
    TTI_SFPNOP;
    TTI_SFPMAD(L2, L5, L6, L3, 0);
    TTI_SFPNOP;
    TTI_SFPMUL(L3, L6, C0, L3, 0);
    TTI_SFPNOP;
    TTI_SFPMAD(L5, L3, L5, L2, 0);  // L2 <- the refined y
    TTI_SFPNOP;
}

// the band at even-slot address `e`: m <- m / (row sum [+ eps]) [+ eps]
FORCE_INLINE void sk_row_normalize(uint32_t e, bool use_eps, bool eps_out) {
    TT_SFPLOAD(L0, IM, ADDR_MOD_7, e);
    TT_SFPLOAD(L1, IM, ADDR_MOD_7, e + ODD);
    TTI_SFPSHFT2(0, L0, L2, ROR1);  // L2[c] = E[c-1]
    TTI_SFPSHFT2(0, L1, L3, ROR1);
    TTI_SFPNOP;
    TTI_SFPADD(L0, C1, L2, L2, 0);  // odd lanes: E[2i] + E[2i+1]
    TTI_SFPADD(L1, C1, L3, L3, 0);
    TTI_SFPNOP;
    TTI_SFPADD(L2, C1, L3, L2, 0);  // odd lanes: the row's sum over its 4 entries
    TTI_SFPNOP;
    TTI_SFPMUL(L2, L4, C0, L2, 0);  // even lanes <- 0
    TTI_SFPNOP;
    sk_ror(L2, L3, 7);              // L3[c] = L2[c+1]: even lane 2i gets the sum from lane 2i+1
    TTI_SFPADD(L2, C1, L3, L2, 0);  // every lane holds its row's sum
    TTI_SFPNOP;
    if (use_eps) {
        TTI_SFPADD(L2, C1, L7, L2, 0);
        TTI_SFPNOP;
    }
    sk_recip_l2();
    TTI_SFPMUL(L0, L2, C0, L0, 0);
    TTI_SFPMUL(L1, L2, C0, L1, 0);
    TTI_SFPNOP;
    if (eps_out) {
        TTI_SFPADD(L0, C1, L7, L0, 0);
        TTI_SFPADD(L1, C1, L7, L1, 0);
        TTI_SFPNOP;
    }
    TT_SFPSTORE(L0, IM, ADDR_MOD_7, e);
    TT_SFPSTORE(L1, IM, ADDR_MOD_7, e + ODD);
}

// one slot (even or odd columns) of a band at address `a`: m <- m / (column sum + eps)
FORCE_INLINE void sk_col_normalize(uint32_t a) {
    TT_SFPLOAD(L0, IM, ADDR_MOD_7, a);
    TTI_SFPNOP;
    sk_ror(L0, L2, 2);              // L2[c] = X[c-2]
    TTI_SFPADD(L0, C1, L2, L2, 0);  // v[c] = X[c] + X[c-2]
    TTI_SFPNOP;
    sk_ror(L2, L3, 4);              // L3[c] = v[c-4]
    TTI_SFPADD(L2, C1, L3, L2, 0);  // X[c] + X[c-2] + X[c-4] + X[c-6]: the column's 4 entries
    TTI_SFPNOP;
    TTI_SFPADD(L2, C1, L7, L2, 0);  // + eps
    TTI_SFPNOP;
    sk_recip_l2();
    TTI_SFPMUL(L0, L2, C0, L0, 0);
    TTI_SFPNOP;
    TT_SFPSTORE(L0, IM, ADDR_MOD_7, a);
}

// Sinkhorn on dest tile `m` (the comb logits' exp), bands 0..nb-1, `iters` iterations, `eps_bits` the fp32 eps:
// m = softmax_row(m) + eps; m /= colsum + eps; then (iters - 1) x { m /= rowsum + eps; m /= colsum + eps } --
// the sinkhorn_reference / mhc_split_sinkhorn sequence, in place.
FORCE_INLINE void sinkhorn_bands(
    uint32_t m, uint32_t /*unused*/, uint32_t /*unused2*/, uint32_t nb, uint32_t iters, uint32_t eps_bits) {
    // constants: L4 = 1.0 on the odd lanes (lane-0 mask rotated to 1, 3, 5, 7), L6 = -1.0, L7 = eps
    sfpu::_build_lane_mask_col0_();  // L6 <- the lane-0 mask (clobbers L2)
    TTI_SFPNOP;
    sk_ror(L6, L4, 1);
    sk_ror(L4, L3, 2);
    TTI_SFPADD(L4, C1, L3, L4, 0);
    TTI_SFPNOP;
    sk_ror(L3, L3, 2);
    TTI_SFPADD(L4, C1, L3, L4, 0);
    TTI_SFPNOP;
    sk_ror(L3, L3, 2);
    TTI_SFPADD(L4, C1, L3, L4, 0);
    TTI_SFPNOP;
    TTI_SFPMOV(0, C0, L6, 0);
    TTI_SFPMOV(0, C0, L7, 0);
    TTI_SFPNOP;
    TTI_SFPLOADI(L6, sfpi::SFPLOADI_MOD0_UPPER, 0xBF80);  // -1.0
    TT_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_LOWER, eps_bits & 0xFFFFu);
    TT_SFPLOADI(L7, sfpi::SFPLOADI_MOD0_UPPER, eps_bits >> 16);
    TTI_SFPNOP;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint32_t e = m * DT + band_off(b);
        sk_row_normalize(e, false, true);  // the row softmax (the exp is in), then + eps
        sk_col_normalize(e);
        sk_col_normalize(e + ODD);
        for (uint32_t i = 1; i < iters; ++i) {
            sk_row_normalize(e, true, false);
            sk_col_normalize(e);
            sk_col_normalize(e + ODD);
        }
    }
}

}  // namespace mhc_sfpu
#endif
