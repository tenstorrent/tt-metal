// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// fill — canonical semantic C++ body (storm contract, fresh_cpp/README.md).
// fill(x) = value for every element; the input is ignored (torch
// Tensor.fill_ semantics, golden_generators._fill with the dispatch
// constant 5.0).  Production: legacy _calculate_fill_ (tt_llk
// ckernel_sfpu_fill.h) broadcasting through vConstFloatPrgm0.  Fresh
// statement: a plain typed constant store; constant residency and delivery
// are the compiler's.
#include <cstdint>

namespace ckernel::sfpu
{

// Fixed dispatch scalar, shared with the golden and identical to the value the
// production dispatch sends (sfpu_operations.h call_unary_sfpu_operation
// fill_const_value default 5.0f; golden_generators._fill const_value=5).
// Both legs must always receive the same value.
constexpr float FRESH_FILL_VALUE = 5.0f;

template <int ITERATIONS>
__attribute__((noinline)) void calculate_fill_fresh_cpp(const float value)
{
    for (int d = 0; d < ITERATIONS; ++d)
    {
        sfpi::dst_reg[0] = sfpi::vFloat(value);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
