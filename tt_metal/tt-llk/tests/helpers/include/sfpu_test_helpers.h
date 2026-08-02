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
        sfpi::dst_reg[0] = sfpu_sqrt_custom<APPROXIMATION_MODE>(sfpi::dst_reg[0]);
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
} // namespace ckernel::sfpu
