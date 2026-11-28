// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_sfpu_remainder.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel {

template <bool APPROXIMATE, DstSync DST_SYNC, bool DST_ACCUM>
__attribute__((always_inline)) inline void llk_math_eltwise_unary_sfpu_remainder_uint32(
    const std::uint32_t dst_index, const std::uint32_t divisor) {
    SFPU_UNARY_CALL(
        DST_SYNC,
        DST_ACCUM,
        calculate_remainder_uint32_scalar,
        (APPROXIMATE /*APPROXIMATION_MODE*/, 8 /*ITERATIONS*/),
        dst_index,
        VectorMode::RC,
        divisor);
}

}  // namespace ckernel
