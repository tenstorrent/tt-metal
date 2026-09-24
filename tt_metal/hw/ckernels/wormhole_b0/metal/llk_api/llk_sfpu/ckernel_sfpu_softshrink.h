// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

inline void softshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_softshrink(std::uint32_t param0) {
    // Softshrink(x) = x - λ if x > λ, x + λ if x < -λ, else 0
    // Algebraically identical to x - clamp(x, -λ, λ)
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = v - sfpi::clamp(v, -lambda, lambda);
        sfpi::dst_reg++;
    }
}

// Op class for softshrink with threshold lambda (param0, fp32 bits).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Softshrink : SfpuUnaryOp<Softshrink<APPROXIMATION_MODE, ITERATIONS>> {
    static constexpr auto& calculate = calculate_softshrink<APPROXIMATION_MODE, ITERATIONS>;
    static inline __attribute__((always_inline)) void init_op() { softshrink_init(); }
};

}  // namespace ckernel::sfpu
