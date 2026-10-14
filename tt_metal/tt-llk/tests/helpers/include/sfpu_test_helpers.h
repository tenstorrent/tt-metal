// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"

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

// Adapts calculate_mask to the 2-tile binary SFPU harness signature. The indices
// are unused: calculate_mask hard-codes its operands (data at dst_reg[0], mask at
// dst_reg[32], result in place), so only the default in0=0/in1=1/out=0 placement works.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_binary(const std::uint32_t /*dst_index_in0*/, const std::uint32_t /*dst_index_in1*/, const std::uint32_t /*dst_index_out*/)
{
    calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
}

// Bit pattern clobber_sfpu_lregs() writes: as a float about 25.7, a value no SFPU kernel programs as a constant.
// Keep it positive and well above 1. A near-zero or negative value (0xABCDABCD is about -1.46e-12) can slip
// through guards: sfpu_reciprocal_iter skips its Newton update when Prgm0 is below about 1, so it fell back to
// the approximate reciprocal and passed the approx-mode contract instead of failing.
constexpr std::uint32_t SFPU_LREG_CLOBBER_BITS = 0x41CDABCDu;

// Writes SFPU_LREG_CLOBBER_BITS to every SFPU register a kernel can write and later read back. Run it just
// before an op's init, where a fused compute kernel would run another op's init: an init that does not program
// a register its body reads then gives wrong results, instead of passing on what reset or an earlier kernel left.
//  - LREG0..7: general purpose.
//  - LREG12..14: the programmable constants (vConstFloatPrgm0..2). SFPCONFIG copies LREG0 into them, so they go
//    first, while LREG0 holds the pattern.
// Left alone: LREG8..10 and LREG15 are read-only (fixed constants, lane id), and LREG11 (LCONST_neg1, -1.0) is a
// core-wide constant any kernel may read without programming it (see _restore_lconst_neg1_).
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
