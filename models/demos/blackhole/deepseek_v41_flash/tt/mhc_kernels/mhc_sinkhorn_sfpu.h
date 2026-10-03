// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// mHC Sinkhorn on the SFPU.  The tile in dst holds, in 32-bit mode, comb element e (= 4*i + j, e < 16) of token t (<
// 32) at
//   face e / 8, vector (e % 8) of that face (= rows 4*((e%8)/2) .. +3 of the face, columns of parity e % 2), lane t,
// i.e. element e is the dst vector number e (sfpi::dst_reg[e]) and the 32 lanes are the 32 tokens: the whole 4x4
// Sinkhorn is lane-wise arithmetic on 16 vectors (no matmul / CB round trips).
//   m = softmax_rows(x) + eps;  m = m / (colsum(m) + eps);  then (iters-1) times: m /= (rowsum + eps); m /= (colsum +
//   eps)
// (x = exp(logits) has already been applied by the caller.)

#if defined(TRISC_MATH)

#include "ckernel_sfpu_recip.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// normalise the 4 vectors A, B, C, D by their sum (+ eps) [+ eps on the result]
template <int IDX, bool EPS_OUT>
sfpi_inline void mhc_sk_scale(const sfpi::vFloat& r, const sfpi::vFloat& veps) {
    sfpi::vFloat x = sfpi::dst_reg[IDX];
    if constexpr (EPS_OUT) {
        sfpi::dst_reg[IDX] = x * r + veps;
    } else {
        sfpi::dst_reg[IDX] = x * r;
    }
}

template <int A, int B, int C, int D, bool EPS_SUM, bool EPS_OUT>
sfpi_inline void mhc_sk_group(const sfpi::vFloat& veps) {
    sfpi::vFloat s = sfpi::dst_reg[A];
    sfpi::vFloat xb = sfpi::dst_reg[B];
    s = s + xb;
    sfpi::vFloat xc = sfpi::dst_reg[C];
    s = s + xc;
    sfpi::vFloat xd = sfpi::dst_reg[D];
    s = s + xd;
    if constexpr (EPS_SUM) {
        s = s + veps;
    }
    const sfpi::vFloat r = sfpu_reciprocal_iter<2>(s);
    mhc_sk_scale<A, EPS_OUT>(r, veps);
    mhc_sk_scale<B, EPS_OUT>(r, veps);
    mhc_sk_scale<C, EPS_OUT>(r, veps);
    mhc_sk_scale<D, EPS_OUT>(r, veps);
}

template <bool EPS_SUM, bool EPS_OUT>
sfpi_inline void mhc_sk_rows(const sfpi::vFloat& veps) {
    mhc_sk_group<0, 1, 2, 3, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<4, 5, 6, 7, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<8, 9, 10, 11, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<12, 13, 14, 15, EPS_SUM, EPS_OUT>(veps);
}

template <bool EPS_SUM, bool EPS_OUT>
sfpi_inline void mhc_sk_cols(const sfpi::vFloat& veps) {
    mhc_sk_group<0, 4, 8, 12, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<1, 5, 9, 13, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<2, 6, 10, 14, EPS_SUM, EPS_OUT>(veps);
    mhc_sk_group<3, 7, 11, 15, EPS_SUM, EPS_OUT>(veps);
}

inline void mhc_sinkhorn_sfpu(std::uint32_t iters, std::uint32_t eps_bits) {
    const sfpi::vFloat veps = Converter::as_float(eps_bits);
    mhc_sk_rows<false, true>(veps);  // row softmax (no eps in the divisor), then + eps
    mhc_sk_cols<true, false>(veps);
    for (std::uint32_t i = 1; i < iters; ++i) {
        mhc_sk_rows<true, false>(veps);
        mhc_sk_cols<true, false>(veps);
    }
}

}  // namespace ckernel::sfpu

#endif
