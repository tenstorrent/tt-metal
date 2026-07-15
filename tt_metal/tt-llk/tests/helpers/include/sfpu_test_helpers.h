// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_sqrt_custom.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"

// Test-only SFPU helpers.
//
// These wrappers exist purely so that production per-vector SFPU primitives can
// be driven through the shared test harness (call_unary_sfpu_operation /
// the 2-tile binary harness). They are not part of any production SFPU API and
// deliberately live outside sfpu_operations.h so that file only contains the
// harness dispatch itself.
namespace ckernel::sfpu
{
// Test-only loop wrapper. The production sqrt_custom header exposes only the
// per-vector helper sfpu_sqrt_custom(); wrap it in the standard dst_reg loop so
// it can be driven through call_unary_sfpu_operation like every other unary op.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sqrt_custom()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = sfpu_sqrt_custom<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Test-only loop wrapper. The expm1_cw header exposes only the per-vector helper
// expm1_cw_clamped() (shared by ELU/CELU/SELU); wrap it in the standard dst_reg
// loop so it can be driven through call_unary_sfpu_operation like every other op.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_expm1_cw()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        sfpi::dst_reg[0] = expm1_cw_clamped(sfpi::dst_reg[0]);
        sfpi::dst_reg++;
    }
}

// Test-only (dst_index_in0, dst_index_in1, dst_index_out) adapter so the float
// mask kernel can be driven through the 2-tile binary SFPU harness. calculate_mask
// hard-codes its operands (data at dst_reg[0], mask at dst_reg[mask_val_idx=32],
// result written in place at dst_reg[0]); those fixed offsets line up with the
// binary harness' tile0/tile1 layout, so the forwarded indices are unused and the
// call only makes sense for the default in0=0/in1=1/out=0 placement.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_binary(const std::uint32_t /*dst_index_in0*/, const std::uint32_t /*dst_index_in1*/, const std::uint32_t /*dst_index_out*/)
{
    calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
}
} // namespace ckernel::sfpu
