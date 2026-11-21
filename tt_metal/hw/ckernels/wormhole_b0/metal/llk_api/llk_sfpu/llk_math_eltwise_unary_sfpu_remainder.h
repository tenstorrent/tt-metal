// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_remainder.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

// Selects the divisor-class kernel here, always inline, so a literal divisor folds the checks and the
// ELF links only the one noinline path the SFPU call wrapper takes the address of.
template <bool APPROXIMATE, DstSync DST_SYNC, bool DST_ACCUM>
__attribute__((always_inline)) inline void llk_math_eltwise_unary_sfpu_remainder_uint32(
    const std::uint32_t dst_index, const std::uint32_t divisor) {
    if (divisor >= 0x80000000u) {
        SFPU_UNARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_remainder_uint32_scalar_big_b,
            (APPROXIMATE /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
            dst_index,
            VectorMode::RC,
            divisor);
    } else if ((divisor & (divisor - 1u)) == 0u) {
        SFPU_UNARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_remainder_uint32_scalar_pow2,
            (APPROXIMATE /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
            dst_index,
            VectorMode::RC,
            divisor);
    } else if (divisor < (1u << 11)) {
        SFPU_UNARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_remainder_uint32_scalar_small_b,
            (APPROXIMATE /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
            dst_index,
            VectorMode::RC,
            divisor);
    } else {
        SFPU_UNARY_CALL(
            DST_SYNC,
            DST_ACCUM,
            calculate_remainder_uint32_scalar_general,
            (APPROXIMATE /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
            dst_index,
            VectorMode::RC,
            divisor);
    }
}

}  // namespace ckernel
