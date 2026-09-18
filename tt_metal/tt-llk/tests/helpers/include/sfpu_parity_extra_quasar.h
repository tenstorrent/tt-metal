// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu/ckernel_sfpu_bitwise.h"
#include "llk_sfpu/ckernel_sfpu_mask.h"
#include "llk_sfpu/ckernel_sfpu_situ_glu.h"

namespace ckernel::sfpu
{

template <bool APPROXIMATION_MODE, UnaryBitwiseOp OP, int ITERATIONS>
inline void calculate_parity_scalar_bitwise()
{
    calculate_sfpu_unary_bitwise<APPROXIMATION_MODE, OP, DataFormat::Int32, ITERATIONS>(0x55u);
}

// The production mask is an in-place tile-0 operation with its mask in tile 1.
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_parity_mask(const uint in0, const uint in1, const uint out)
{
    LLK_ASSERT(in0 == 0 && in1 == 1 && out == 0, "Parity mask requires data/mask/output tiles 0/1/0");
    calculate_mask<APPROXIMATION_MODE, ITERATIONS>();
}

} // namespace ckernel::sfpu
