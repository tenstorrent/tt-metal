// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the rdiv op (storm contract,
// fresh_cpp/README.md).  Production: metal ckernel_sfpu_rdiv.h
// calculate_rdiv — sfpu_reciprocal_iter reading its Newton constant 2.0
// from vConstFloatPrgm0 (programmed by rdiv_init), RoundingMode dispatch,
// unroll-8 pin.  rdiv(x) = value / x stated over the shared literal-constant
// reciprocal (fresh_cpp/helpers.h fresh_recip — the same
// approx_recip-plus-guarded-Newton algorithm).  The bf16-dest contract
// (the measured row's variant, is_fp32_dest_acc_en = false) rounds the
// RECIPROCAL to bf16 before the multiply, exactly as production does: the
// numerator 2.0 is a power of two, so the product stays bf16-representable
// and the SFPSTORE truncation is exact.
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Fixed dispatch numerator shared with the golden and the production
// dispatch (sfpu_operations.h 0x40000000u == golden_generators value=2.0).
constexpr float FRESH_RDIV_VALUE = 2.0f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_rdiv_fresh_cpp(const float value)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat y       = fresh_recip<1>(x);
        y                    = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg[0]     = y * value;
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
