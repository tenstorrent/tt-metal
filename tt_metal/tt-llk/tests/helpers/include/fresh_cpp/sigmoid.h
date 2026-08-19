// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the sigmoid op (storm contract, migrated
// verbatim from fresh_cpp_operations.h, Lane BR batch 3).
#include <cstdint>

#include "fresh_cpp/helpers.h"

namespace ckernel::sfpu
{

// Sigmoid, bf16 non-approx contract (production: _sfpu_sigmoid_ spread across
// three headers — exp_21f helper + sfpu_reciprocal_iter<1> reading its 2.0
// from vConstFloatPrgm0).  sigmoid(x) = 1/(1 + exp(-x)) stated in one place,
// every constant local.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_sigmoid_fresh_cpp()
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x = sfpi::dst_reg[0];
        const sfpi::vFloat e = fresh_exp_21f(-x);
        const sfpi::vFloat y = fresh_recip<1>(1.0f + e);
        sfpi::dst_reg[0]     = sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
