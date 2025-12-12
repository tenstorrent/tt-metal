// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2025 Jason Davies <jason@jasondavies.com>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <climits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_isinf_isnan.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// computes L1=trunc(L0).
inline void _trunc_body_()
{
    // set L3=23.  TODO: this could be stored in a constant register, but use by rdiv prevents this for now.
    TTI_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_SHORT, 23);
    // mask = 0x8000_0000
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_FLOATB, 0x8000);
    // disable lanes where exp < 0
    TTI_SFPEXEXP(0, p_sfpu::LREG0, p_sfpu::LREG2, sfpi::SFPEXEXP_MOD1_SET_CC_SGN_EXP | sfpi::SFPEXEXP_MOD1_SET_CC_COMP_EXP);
    // mask = 0xffff_ffff
    TTI_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_SHORT, 0xffff);
    // exp = 23 - exp
    TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG2, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_GTE0);
    // mask <<= exp
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG2, p_sfpu::LREG1, sfpi::SFPSHFT2_MOD1_SHFT_LREG);
    // reset lanes
    TTI_SFPENCC(0, 0, 0, 0);
    // apply mask
    TTI_SFPAND(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
}

// computes L1=floor(L0).
inline void _floor_body_()
{
    _trunc_body_();
    // if v>u, set v=v-1; this only happens for negative values.
    // on Wormhole, we don't have SFPGT, so use u<0 and (v-u)<0 instead.
    // First, ensure u<0.
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_LT0);
    // Then, ensure (v-u)<0 (two's complement).
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

// computes L1=ceil(L0).
inline void _ceil_body_()
{
    _trunc_body_();
    // if v<u, set v=v+1.
    // on Wormhole, we don't have SFPGT, so use u>=0 and (v-u)<0 instead.
    // First, ensure u>=0.
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_GTE0);
    // Then, ensure (v-u)<0 (two's complement).
    TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPIADD_MOD1_ARG_2SCOMP_LREG_DST | sfpi::SFPIADD_MOD1_CC_LT0);
    TTI_SFPMAD(p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG1, 0);
    TTI_SFPENCC(0, 0, 0, 0);
}

inline constexpr std::array<float, 84> PRECOMPUTED_POW10_TABLE = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F, 1e-31F, 1e-30F, 1e-29F,
    1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F, 1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F,
    1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,  1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,
    1e6F,   1e7F,   1e8F,   1e9F,   1e10F,  1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,
    1e23F,  1e24F,  1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_floor_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _floor_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_ceil_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _ceil_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_trunc_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _trunc_body_();
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _calculate_frac_()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
        _trunc_body_();
        // frac(x) = x - trunc(x)
        TTI_SFPMAD(p_sfpu::LREG1, p_sfpu::LCONST_neg1, p_sfpu::LREG0, p_sfpu::LREG1, 0);
        TTI_SFPNOP;
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

inline sfpi::vFloat _round_even_(sfpi::vFloat v)
{
    // Create a temporary copy tmp = abs(v).
    sfpi::vFloat tmp = sfpi::setsgn(v, 0);
    // For all 0 ≤ x < 2**23, x + 2**23 will shift out the fractional part with round-to-nearest-even.
    tmp += 8388608.0f;
    // Hide SFPNOP; extract exponent.
    sfpi::vInt exp = sfpi::exexp(v);
    // Subtract 2**23 to restore exponent.
    tmp += -8388608.0f;
    // Hide SFPNOP; check exponent.  If x ≥ 2**23, then there is no fractional part.
    v_if (exp < 23)
    {
        // v.{Exp,Man}=tmp.{Exp,Man}; retaining original sign.
        v = sfpi::setsgn(tmp, v);
    }
    v_endif;
    return v;
}

template <bool APPROXIMATE, int ITERATIONS = 8>
void _calculate_round_(const int decimals)
{
    const auto exp10i = [](int n)
    {
        if (n > 38) // 38 is max decimal places float32 can store for positive values
        {
            return 1.0F / 0.0F;
        }

        if (n < -45) // 45 is max decimal places float32 can store for negative values
        {
            return 0.0F;
        }

        return PRECOMPUTED_POW10_TABLE[n + 45];
    };

    const sfpi::vFloat coeff   = exp10i(decimals);
    const sfpi::vFloat inverse = exp10i(-decimals);

    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::vFloat v      = sfpi::dst_reg[0];
        sfpi::vFloat result = inverse * _round_even_(v * coeff);
        sfpi::dst_reg[0]    = result;
        sfpi::dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
