// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_sfpu/ckernel_sfpu_expm1_cw.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"

// Test-only wrappers that adapt production per-vector SFPU primitives to the
// shared test harness. Not part of any production SFPU API; kept out of
// sfpu_operations.h so that file only holds harness dispatch.
namespace ckernel::sfpu
{
// Wraps the per-vector sfpu_sqrt_custom() in the standard dst_reg loop so it can
// run through call_unary_sfpu_operation.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sqrt_custom()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        // This wrapper is the only caller that can be handed a -inf, and the op's IEEE
        // contract is what the edge sweep checks, so it pays for the guard that erfinv and
        // asin/acos do not. See ckernel_sfpu_sqrt_custom.h for the measured cost.
        sfpi::dst_reg[0] = sfpu_sqrt_custom<APPROXIMATION_MODE, 2, true /*NEGATIVE_INFINITY_SAFE*/>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Wraps the per-vector expm1_cw_clamped() (shared by ELU/CELU/SELU) in the
// standard dst_reg loop so it can run through call_unary_sfpu_operation.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_expm1_cw()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = expm1_cw_clamped(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Adapts calculate_mask / calculate_mask_posinf to the 2-tile binary SFPU harness signature.
// The indices are unused: both hard-code their operands (data at dst_reg[0], mask at
// dst_reg[32], result in place), so only the default in0=0/in1=1/out=0 placement works.
// POSINF selects calculate_mask_posinf, where the data becomes +inf where the mask is zero,
// over calculate_mask, where it becomes 0.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool POSINF = false>
inline void calculate_mask_binary(const std::uint32_t /*dst_index_in0*/, const std::uint32_t /*dst_index_in1*/, const std::uint32_t /*dst_index_out*/)
{
    if constexpr (POSINF)
    {
        calculate_mask_posinf<APPROXIMATION_MODE, ITERATIONS>();
    }
    else
    {
        calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
    }
}

// 0x42 repeated (about 48.56 as a float): easy to spot in dumps. Keep it positive and well above 1, or guards like
// sfpu_reciprocal_iter's skip the Newton step and a broken init still passes.
constexpr std::uint32_t SFPU_LREG_CLOBBER_BITS = 0x42424242u;

// Fills LREG0..7 and the programmable constants LREG12..14 (via LREG0, so those go first) with the pattern.
// LREG11 (-1.0) is left alone on purpose: kernels may read it without programming it.
inline void clobber_sfpu_lregs()
{
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, SFPU_LREG_CLOBBER_BITS >> 16);
    TTI_SFPLOADI(p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, SFPU_LREG_CLOBBER_BITS & 0xFFFF);
    TTI_SFPCONFIG(0, p_sfpu::LREG12, 0);
    TTI_SFPCONFIG(0, p_sfpu::LREG13, 0);
    TTI_SFPCONFIG(0, p_sfpu::LREG14, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG5, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG6, 0);
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG7, 0);
}
} // namespace ckernel::sfpu
