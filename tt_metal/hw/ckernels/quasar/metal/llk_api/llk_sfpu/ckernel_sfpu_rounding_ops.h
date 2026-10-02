// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// Trunc/floor/ceil are TTI because SFPI currently produces slower kernels. Macro signatures
// dropped unused SFPU fields (SFPENCC 2-arg, SFPAND 2-arg, SFPEXEXP 3-arg); do not paste BH 4-arg
// forms. Encodings live in ckernel_instr_params.h (p_sfploadi / p_sfpexexp / p_sfpiadd / p_sfpshft2).
//
// Predication is LaneEnabled = (~CC.En | CC.Res). The trunc chain only *narrows* (never re-enables
// mid-body), so LaneEnabled-gated CC writes on SFPEXEXP / SFPIADD / SFPGT are required, not a
// reason to avoid TTI. CC is already on: MATH TRISC firmware (`trisc.cc` `enable_cc_stack`) and the
// LLK `boot.h` setup both issue `TTI_SFPENCC(3, 10)`. Same assumption as BH rounding.

// Truncate toward zero by CC-overlapped mantissa masks (same lane outcomes as Blackhole):
//   exp < 0  (|x| < 1, including ±0 and subnormals): keep sign bit only → ±0
//   0 ≤ exp ≤ 23: shift 0xffffffff left by (23 − exp) to drop fractional mantissa bits
//   exp > 23 (already integral, inf, nan): keep all bits
sfpi_inline sfpi::vFloat _trunc_body_(sfpi::vFloat val) {
    sfpi::l_reg[sfpi::LRegs::LReg0] = val;
    TTI_SFPLOADI(p_sfpu::LREG3, p_sfploadi::MOD0_INT16, 23);
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);  // 0x80000000
    TTI_SFPEXEXP(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpexexp::MOD1_SET_CC_GE0);
    TTI_SFPLOADI(p_sfpu::LREG1, p_sfploadi::MOD0_INT16, 0xffff);  // 0xffffffff on remaining lanes
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpiadd::MOD1_SUB_CC_GTE0);
    TTI_SFPSHFT2(0, p_sfpu::LREG2, p_sfpu::LREG1, p_sfpshft2::MOD1_SHFT_LREG);
    TTI_SFPENCC(0, 0);
    TTI_SFPAND(p_sfpu::LREG0, p_sfpu::LREG1);  // LREG1 &= LREG0

    sfpi::l_reg[sfpi::LRegs::LReg2].in_use();
    sfpi::l_reg[sfpi::LRegs::LReg3].in_use();

    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

// Floor: trunc, then trunc−1 on lanes where val < trunc (negative non-integers).
sfpi_inline sfpi::vFloat _floor_body_(sfpi::vFloat val) {
    sfpi::l_reg[sfpi::LRegs::LReg1] = _trunc_body_(val);
    TTI_SFPGT(p_sfpgt::IMM12_FP32, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpgt::MOD1_SET_CC);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0);
    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

// Ceil: trunc, then trunc+1 on lanes where val > trunc (positive non-integers).
sfpi_inline sfpi::vFloat _ceil_body_(sfpi::vFloat val) {
    sfpi::l_reg[sfpi::LRegs::LReg1] = _trunc_body_(val);
    TTI_SFPGT(p_sfpgt::IMM12_FP32, p_sfpu::LREG1, p_sfpu::LREG0, p_sfpgt::MOD1_SET_CC);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0);
    return sfpi::l_reg[sfpi::LRegs::LReg1];
}

// Round-to-nearest-even for |x| < 2^23 via the 2^23 magic add; larger magnitudes are already
// integral. Uses abs + copysgn so the add is performed on a non-negative value (same as BH).
sfpi_inline sfpi::vFloat _round_even_(sfpi::vFloat v) {
    sfpi::vFloat tmp = sfpi::setsgn(v, 0);
    tmp += 0x1.p23f;
    const sfpi::vInt exp = sfpi::exexp(v);
    tmp += -0x1.p23f;
    v_if(exp < 23) { v = sfpi::copysgn(tmp, v); }
    v_endif;
    return v;
}

inline constexpr std::array<float, 84> PRECOMPUTED_POW10_TABLE = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F,
    1e-31F, 1e-30F, 1e-29F, 1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F,
    1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F, 1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,
    1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,   1e6F,   1e7F,   1e8F,   1e9F,   1e10F,
    1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,  1e23F,  1e24F,
    1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_floor_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _floor_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_ceil_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _ceil_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_trunc_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::dst_reg[0] = _trunc_body_(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_frac_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = x - _trunc_body_(x);
        sfpi::dst_reg++;
    }
}

// Scale by 10^decimals, round-even to integer, scale back. Template name APPROXIMATION_MODE
// (BH used APPROXIMATE) so the compute-API pack (APPROX, ITERATIONS) matches other unaries.
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_round_(const int decimals) {
    const auto exp10i = [](int n) {
        if (n > 38) {
            return 1.0F / 0.0F;
        }
        if (n < -45) {
            return 0.0F;
        }
        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coeff = exp10i(decimals);
    const sfpi::vFloat inverse = exp10i(-decimals);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        const sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = inverse * _round_even_(v * coeff);
        sfpi::dst_reg++;
    }
}

// Stochastic FP32 → BF16 narrow. Matches the Blackhole compute-API signature.
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void _calculate_stochastic_round_() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        x = sfpi::convert<sfpi::vFloat16b>(x, sfpi::RoundMode::NearestStochastic);
        sfpi::dst_reg[0] = x;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
