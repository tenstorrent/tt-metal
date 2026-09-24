// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

inline void hardshrink_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(std::uint32_t param0) {
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison using abs: setsgn(v, 0) clears sign bit
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    sfpi::vFloat lambda = Converter::as_float(param0);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];
        sfpi::vFloat abs_v = sfpi::setsgn(v, 0);
        v_if(abs_v <= lambda) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        sfpi::dst_reg++;
    }
}

// Op class for hardshrink with threshold lambda (param0, fp32 bits).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Hardshrink : SfpuUnaryOp<Hardshrink<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t param0) {
        calculate_hardshrink<APPROXIMATION_MODE, ITERATIONS>(param0);
    }
    static inline __attribute__((always_inline)) void init_op() { hardshrink_init(); }
};

}  // namespace ckernel::sfpu
