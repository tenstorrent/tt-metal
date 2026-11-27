// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, vec: hardtanh (llk_lib/vec/common.h has the shape of a function file).
// min(max(x, min), max). Moved from tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_hardtanh.h.

#pragma once

#include <cstdint>

#include "llk_lib/vec/common.h"

#ifdef LLK2_SFPU_THREAD

namespace llk2::llk_lib
{

/// hardtanh's part of vec_configure: nothing of its own. Fast and Fp32 are not read.
template <bool Fast, bool Fp32>
inline void vec_hardtanh_init()
{
}

// Hardtanh(x) = max_val if x > max_val, min_val if x < min_val, else x
// Equivalent to: clamp(x, min_val, max_val) = min(max(x, min_val), max_val)
// The two bounds are FP32 bit patterns. One form: Fast and Fp32 are not read.
template <bool Fast, bool Fp32, int Iterations>
inline void vec_hardtanh_iterations(std::uint32_t min_bits, std::uint32_t max_bits)
{
    // Materialize both bounds outside the loop for better performance
    sfpi::vFloat min_val = vec_float_of(min_bits);
    sfpi::vFloat max_val = vec_float_of(max_bits);
#pragma GCC unroll 8
    for (int d = 0; d < Iterations; d++)
    {
        sfpi::vFloat v   = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::clamp(v, min_val, max_val);
        sfpi::dst_reg++;
    }
}

} // namespace llk2::llk_lib

#endif // LLK2_SFPU_THREAD
