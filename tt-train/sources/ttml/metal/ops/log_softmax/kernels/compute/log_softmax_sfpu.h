// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
namespace ckernel::sfpu {
// All values and indices remain FP32. Column indices are exact through 2^24.
template <int mode, int index_offset>
inline void calculate_log_softmax_mask(uint32_t width) {
    const sfpi::vFloat limit = static_cast<float>(width);
#pragma GCC unroll 8
    for (int i = 0; i < 8; ++i) {
        sfpi::vFloat value = sfpi::dst_reg[0];
        sfpi::vFloat index = sfpi::dst_reg[index_offset * 32];
        if constexpr (mode == 0) {
            v_if(index >= limit) {
                value = -std::numeric_limits<float>::infinity();
            }
            v_endif;
        } else if constexpr (mode == 1) {
            v_if(sfpi::abs(value) != 0.0f) {
                index = std::numeric_limits<float>::infinity();
            }
            v_endif;
            v_if(index >= limit) {
                index = std::numeric_limits<float>::infinity();
            }
            v_endif;
            value = index;
        } else if constexpr (mode == 2) {
            v_if(index >= limit) {
                value = 0.0f;
            }
            v_endif;
        } else if constexpr (mode == 3) {
            v_if(sfpi::abs(index) == 0.0f) {
                value = 0.0f;
            }
            v_endif;
        } else {
            const sfpi::vFloat replacement = sfpi::dst_reg[64];
            v_if(sfpi::abs(index) == 0.0f) {
                value = replacement;
            }
            v_endif;
        }
        sfpi::dst_reg[0] = value;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu
#endif
namespace ckernel {
template <int mode, int index_offset = 1>
ALWI void log_softmax_mask(uint32_t dst, uint32_t width) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_log_softmax_mask, (mode, index_offset), dst, VectorMode::RC, width));
}
}  // namespace ckernel
