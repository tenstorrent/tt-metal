// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// chunk_gated_delta_rule (prep) only: the per-row inverse RMS of a Gram tile, as a diagonal tile, in ONE pass.
// DST[x] holds G = X @ X^T (row sums of squares on the diagonal), DST[eye] the 32x32 identity. Result in place:
//   D = eye * ([scale *] rsqrt(G * eye + eps))
// which is exactly the five-call chain mul_binary(eye), add_unary(eps), rsqrt, [mul_unary(scale)], mul_binary(eye)
// -- the same fp32 ops in the same order, the same rsqrt body -- but run only on faces 0 and 3 (the faces holding
// the diagonal). Faces 1 and 2 are pure off-diagonal: the chain gives +0 there (eye = 0 before the rsqrt makes it
// rsqrt(eps) * 0), so they are stored as +0 directly.
// Absolute DST addressing (tile * 32 sfpi rows, face * 8); call once with the DST base at 0 and a fresh RWC.
// One face (8 sfpi rows) of the diagonal, x at sfpi row x0, the identity at e0, with ONE rsqrt body for the face.
// One sfpi row is four tile rows by eight columns of one face (the even columns at an even DEST address, the odd ones
// at +2; lane = row_in_group * 8 + col / 2, see ckernel_sfpu_triangle_solve.h), so the face's 16 diagonal entries sit
// in 16 DISTINCT lanes of its 8 sfpi rows: (i, i) is row (i / 4) * 2 + (i & 1), lane (i & 3) * 8 + i / 2. The masked
// rows G_d * eye_d summed over d therefore hold every diagonal entry in its own lane, exactly (the other terms are
// +-0); the rsqrt body runs once on that vector, and each row is rebuilt as r * eye_d, which selects its diagonal
// lanes. Every diagonal lane goes through the same fp32 ops as the per-row form (G * 1 + eps, rsqrt, [* scale],
// * 1) and every other lane comes out +0 as before (a finite positive r times 0), so D is bit-identical to it.
// Fully unrolled with immediate DEST offsets: no RWC moves.
template <bool APPROXIMATION_MODE, bool DO_SCALE>
inline void gdn_inv_rms_diag_face(
    const std::uint32_t x0, const std::uint32_t e0, const std::uint32_t eps_bits, const std::uint32_t scale_bits) {
    sfpi::vFloat v = sfpi::dst_reg[x0] * sfpi::dst_reg[e0];
#pragma GCC unroll 7
    for (std::uint32_t d = 1; d < 8; d++) {
        v = sfpi::dst_reg[x0 + d] * sfpi::dst_reg[e0 + d] + v;
    }
    v = v + Converter::as_float(eps_bits);
    v = _calculate_sqrt_body_<APPROXIMATION_MODE, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(v);
    if constexpr (DO_SCALE) {
        v = v * Converter::as_float(scale_bits);
    }
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < 8; d++) {
        sfpi::dst_reg[x0 + d] = v * sfpi::dst_reg[e0 + d];
    }
}

template <bool APPROXIMATION_MODE, bool DO_SCALE>
inline void calculate_gdn_inv_rms_diag(
    const std::uint32_t dst_x,
    const std::uint32_t dst_eye,
    const std::uint32_t eps_bits,
    const std::uint32_t scale_bits) {
    constexpr std::uint32_t kTile = 32;  // sfpi rows per 32x32 tile
    constexpr std::uint32_t kFace = 8;   // sfpi rows per 16x16 face
    const std::uint32_t x0 = dst_x * kTile;
    const std::uint32_t e0 = dst_eye * kTile;
    // face 0 (rows 0..15, cols 0..15)
    gdn_inv_rms_diag_face<APPROXIMATION_MODE, DO_SCALE>(x0, e0, eps_bits, scale_bits);
    // faces 1 and 2 (off-diagonal) -> +0
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < 2 * kFace; d++) {
        sfpi::dst_reg[x0 + kFace + d] = 0.0f;
    }
    // face 3 (rows 16..31, cols 16..31)
    gdn_inv_rms_diag_face<APPROXIMATION_MODE, DO_SCALE>(x0 + 3 * kFace, e0 + 3 * kFace, eps_bits, scale_bits);
}

}  // namespace ckernel::sfpu
