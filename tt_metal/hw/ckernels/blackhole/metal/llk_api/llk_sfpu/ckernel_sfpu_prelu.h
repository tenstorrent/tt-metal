// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void prelu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(const std::uint32_t value) {
    // SFPU microcode
    vFloat init = Converter::as_float(value);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * init; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

// Op class for prelu with a scalar slope (fp32 bits).
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
struct Prelu : SfpuUnaryOp<Prelu<APPROXIMATION_MODE, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t value) {
        calculate_prelu<APPROXIMATION_MODE, ITERATIONS>(value);
    }
    static inline __attribute__((always_inline)) void init_op() { prelu_init(); }
};
}  // namespace sfpu
}  // namespace ckernel
