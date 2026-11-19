// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// mHC coefficients of up to 32 tokens in ONE SFPU pass over a DST tile in the plane layout (mhc_pre_common.hpp
// plane_off: sfpi::dst_reg[p] lane t = value p of token t). Input planes: the logits z (already m * scale + bias):
//   0..3 pre, 4..7 post, 8..23 comb (row j, column k at 8 + 4j + k).
// Output, in place:
//   pre_i   = sigmoid(z_i) + eps
//   post_i  = 2 sigmoid(z_{4+i})
//   comb    = softmax(z_comb, dim=-1) + eps, then comb /= (column sums + eps), then iters - 1 times
//             { comb /= (row sums + eps); comb /= (column sums + eps) }
// (the fused_pre_post + sinkhorn_fp32 arithmetic of the op chain: row max subtracted before exp, plain row sum in the
// softmax denominator, eps added to every normalising sum). exp is the library's fp32-accurate exp
// (_sfpu_exp_fp32_accurate_, literal constants only); 1/x is the hardware approximation refined by two Newton steps
// with a literal 2.0 (no programmable constant registers are touched, so there is no init ordering against the other
// SFPU ops of the kernel). Every DST index is a template constant (SFPLOAD / SFPSTORE take immediate addresses).
#pragma once

#ifdef TRISC_MATH
#include <utility>

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vFloat mhc_recip(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    sfpi::vFloat t = 2.0f - x * y;
    y = y * t;
    t = 2.0f - x * y;
    y = y * t;
    return y;
}

sfpi_inline sfpi::vFloat mhc_sigmoid(const sfpi::vFloat z) {
    // exp(-z) is clamped below its fp32 overflow (z < -88.7 would give inf, then inf * 0 = NaN in the Newton step):
    // sigmoid is then ~6e-39, i.e. 0 in every consumer
    const sfpi::vFloat e = _sfpu_exp_fp32_accurate_<false>(sfpi::min(-z, 88.0f));
    return mhc_recip(e + 1.0f);
}

// plane P <- sigmoid(plane P) + eps
template <int P, uint32_t EPS_BITS>
sfpi_inline void mhc_pre_plane() {
    const sfpi::vFloat z = sfpi::dst_reg[P];
    sfpi::dst_reg[P] = mhc_sigmoid(z) + Converter::as_float(EPS_BITS);
}

// plane P <- 2 sigmoid(plane P)
template <int P>
sfpi_inline void mhc_post_plane() {
    const sfpi::vFloat z = sfpi::dst_reg[P];
    sfpi::dst_reg[P] = mhc_sigmoid(z) * 2.0f;
}

// the four planes A, B, C, D of one comb row (or column) <- each / (A + B + C + D + eps). The operands are
// re-loaded from DST instead of being kept live (the SFPU has 8 LREGs and the reciprocal needs 3).
template <int A, int B, int C, int D, uint32_t EPS_BITS>
sfpi_inline void mhc_norm4() {
    sfpi::vFloat s = sfpi::dst_reg[A];
    s = s + sfpi::dst_reg[B];
    s = s + sfpi::dst_reg[C];
    s = s + sfpi::dst_reg[D];
    const sfpi::vFloat r = mhc_recip(s + Converter::as_float(EPS_BITS));
    sfpi::dst_reg[A] = sfpi::dst_reg[A] * r;
    sfpi::dst_reg[B] = sfpi::dst_reg[B] * r;
    sfpi::dst_reg[C] = sfpi::dst_reg[C] * r;
    sfpi::dst_reg[D] = sfpi::dst_reg[D] * r;
}

constexpr int MHC_SCRATCH = 24;  // planes 24..31 are free

// plane P <- exp(plane P - plane MHC_SCRATCH)
template <int P>
sfpi_inline void mhc_exp_shifted() {
    const sfpi::vFloat a = sfpi::dst_reg[P];
    const sfpi::vFloat v = a - sfpi::dst_reg[MHC_SCRATCH];
    sfpi::dst_reg[P] = _sfpu_exp_fp32_accurate_<false>(v);
}

// softmax over the comb row J (planes 8 + 4J .. 8 + 4J + 3), + eps; the row max is parked in plane 24
template <int J, uint32_t EPS_BITS>
sfpi_inline void mhc_softmax_row() {
    constexpr int P = 8 + 4 * J;
    {
        sfpi::vFloat mx = sfpi::max(sfpi::vFloat(sfpi::dst_reg[P + 0]), sfpi::vFloat(sfpi::dst_reg[P + 1]));
        mx = sfpi::max(mx, sfpi::vFloat(sfpi::dst_reg[P + 2]));
        mx = sfpi::max(mx, sfpi::vFloat(sfpi::dst_reg[P + 3]));
        sfpi::dst_reg[MHC_SCRATCH] = mx;
    }
    mhc_exp_shifted<P + 0>();
    mhc_exp_shifted<P + 1>();
    mhc_exp_shifted<P + 2>();
    mhc_exp_shifted<P + 3>();
    sfpi::vFloat s = sfpi::dst_reg[P + 0];
    s = s + sfpi::dst_reg[P + 1];
    s = s + sfpi::dst_reg[P + 2];
    s = s + sfpi::dst_reg[P + 3];
    const sfpi::vFloat r = mhc_recip(s);
    const sfpi::vFloat eps = Converter::as_float(EPS_BITS);
    sfpi::dst_reg[P + 0] = sfpi::dst_reg[P + 0] * r + eps;
    sfpi::dst_reg[P + 1] = sfpi::dst_reg[P + 1] * r + eps;
    sfpi::dst_reg[P + 2] = sfpi::dst_reg[P + 2] * r + eps;
    sfpi::dst_reg[P + 3] = sfpi::dst_reg[P + 3] * r + eps;
}

// row pass (dim=-1): comb[j][:] /= (sum_k comb[j][k] + eps)
template <uint32_t EPS_BITS>
sfpi_inline void mhc_row_pass() {
    mhc_norm4<8, 9, 10, 11, EPS_BITS>();
    mhc_norm4<12, 13, 14, 15, EPS_BITS>();
    mhc_norm4<16, 17, 18, 19, EPS_BITS>();
    mhc_norm4<20, 21, 22, 23, EPS_BITS>();
}

// column pass (dim=-2): comb[:][k] /= (sum_j comb[j][k] + eps)
template <uint32_t EPS_BITS>
sfpi_inline void mhc_col_pass() {
    mhc_norm4<8, 12, 16, 20, EPS_BITS>();
    mhc_norm4<9, 13, 17, 21, EPS_BITS>();
    mhc_norm4<10, 14, 18, 22, EPS_BITS>();
    mhc_norm4<11, 15, 19, 23, EPS_BITS>();
}

// Stage 14 (MHC_PRE_EARLY): the 4 pre planes only (the K cores' collapse needs nothing else), ahead of the Sinkhorn
template <uint32_t EPS_BITS>
inline void mhc_pre_planes() {
    mhc_pre_plane<0, EPS_BITS>();
    mhc_pre_plane<1, EPS_BITS>();
    mhc_pre_plane<2, EPS_BITS>();
    mhc_pre_plane<3, EPS_BITS>();
}

template <uint32_t ITERS, uint32_t EPS_BITS>
inline void mhc_coeff_planes() {
    mhc_pre_plane<0, EPS_BITS>();
    mhc_pre_plane<1, EPS_BITS>();
    mhc_pre_plane<2, EPS_BITS>();
    mhc_pre_plane<3, EPS_BITS>();
    mhc_post_plane<4>();
    mhc_post_plane<5>();
    mhc_post_plane<6>();
    mhc_post_plane<7>();
    mhc_softmax_row<0, EPS_BITS>();
    mhc_softmax_row<1, EPS_BITS>();
    mhc_softmax_row<2, EPS_BITS>();
    mhc_softmax_row<3, EPS_BITS>();
    mhc_col_pass<EPS_BITS>();
#pragma GCC unroll 0
    for (uint32_t it = 1; it < ITERS; ++it) {
        mhc_row_pass<EPS_BITS>();
        mhc_col_pass<EPS_BITS>();
    }
}

// Layout probe (MHC_PLANE_PROBE): plane p <- 1000 * plane p + p. The root writes the result to the debug taps; the
// host checks that every element moved by exactly its plane_off plane index (a wrong lane / column-parity assumption
// shows up as elements carrying a foreign plane's offset).
template <int P>
sfpi_inline void mhc_probe_plane() {
    const sfpi::vFloat v = sfpi::dst_reg[P];
    sfpi::dst_reg[P] = v * 1000.0f + static_cast<float>(P);
}

template <int... Ps>
sfpi_inline void mhc_probe_planes(std::integer_sequence<int, Ps...>) {
    (mhc_probe_plane<Ps>(), ...);
}

inline void mhc_plane_probe() { mhc_probe_planes(std::make_integer_sequence<int, 32>{}); }

}  // namespace ckernel::sfpu
#endif  // TRISC_MATH
