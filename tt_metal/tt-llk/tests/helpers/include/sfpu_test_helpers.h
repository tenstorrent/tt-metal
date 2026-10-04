// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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
} // namespace ckernel::sfpu
