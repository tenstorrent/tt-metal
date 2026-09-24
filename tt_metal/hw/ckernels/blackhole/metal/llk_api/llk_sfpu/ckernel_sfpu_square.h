// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_conversions.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

inline void square_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_square() {
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat result = v * v;
        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Op class for x * x.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
struct Square : SfpuUnaryOp<Square<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>> {
    static constexpr auto& calculate = calculate_square<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op() { square_init(); }
};

}  // namespace ckernel::sfpu
