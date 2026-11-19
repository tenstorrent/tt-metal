// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK attention rev 8 (attn_common.hpp): the membership mask applied on the SFPU from the expanded operand.
// DST tile `in1` holds T = B_col @ E8 (the FPU product of the mask operand -- column 0 = the row's membership byte
// as an integer -- with the E8 constant whose row 0 is 2^-s on pool s's four columns): T[r][c] = byte_r x 2^-(c / 4),
// exact in fp32. Pool s is a member of row r iff bit s of byte_r is set iff floor(T) is odd. One pass over the 32
// rows of the tile (4 faces x 8 sfpi rows, VectorMode::RC through the binary-SFPU params wrapper): S = S x scale
// (the fused scale multiply, the same fp32 SFPMUL as mul_unary_tile) and S += -1e9 (the exact fp32 constant of the
// rev 4..6 masks, NEG_MASK_BITS, loaded as 32-bit integer bits) where the key is masked. Masked keys thus tie and
// exp(S - m') underflows to 0 exactly: the output is the rev 6 / 7 output (README §9a).
#pragma once

#ifdef TRISC_MATH
#include "ckernel_sfpu_conversions.h"
#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu {
// (in0 = the S tile, in1 = the T tile, out unused: in place on in0; scale_bits = the fp32 bits of the QK scale)
inline void pmk_mask_apply(
    const std::uint32_t dst_s, const std::uint32_t dst_t, const std::uint32_t /*dst_out*/, std::uint32_t scale_bits) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    const sfpi::vFloat neg = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(0xCE6E6B28u)));  // -1e9f
    const sfpi::vFloat scale = sfpi::as<sfpi::vFloat>(sfpi::vInt(static_cast<int>(scale_bits)));
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat s = sfpi::dst_reg[dst_s * dst_tile_size_sfpi];
        const sfpi::vFloat t = sfpi::dst_reg[dst_t * dst_tile_size_sfpi];
        s = s * scale;
        const sfpi::vInt k = _float_to_int32_positive_(t);
        v_if((k & 1) == 0) { s = s + neg; }
        v_endif;
        sfpi::dst_reg[dst_s * dst_tile_size_sfpi] = s;
        sfpi::dst_reg++;
    }
}

// Rev 10 (PMK_ATT_EXP_21F): E = exp(x) on the exp_21f algorithm in fp32 DST (ckernel_sfpu_exp.h
// _sfpu_exp_21f_bf16_<true>: ~2^-8 relative, the bf16 resolution the P V operand rounds E to anyway; inputs < -88.5
// give 0, so the masked keys' -1e9 vanish exactly). In place on `dst`; the running-max rescale a = exp(m - m') keeps
// the accurate exp (a multiplies the whole partial every rescale: README §12).
inline void pmk_exp21f(const std::uint32_t dst, const std::uint32_t /*unused*/, const std::uint32_t /*unused*/) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        const sfpi::vFloat x = sfpi::dst_reg[dst * dst_tile_size_sfpi];
        sfpi::dst_reg[dst * dst_tile_size_sfpi] = _sfpu_exp_21f_bf16_<true>(x);
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
