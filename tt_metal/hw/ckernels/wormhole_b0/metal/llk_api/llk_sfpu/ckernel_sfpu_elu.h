// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

inline void elu_init() { math::reset_counters(p_setrwc::SET_ABD_F); }

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_elu(std::uint32_t slope) {
    sfpi::vFloat alpha = Converter::as_float(slope);
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * expm1_cw_clamped(x);

        v_if(x >= 0.0f) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// Op class for elu with slope alpha (fp32 bits).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
struct Elu : SfpuUnaryOp<Elu<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t slope) {
        calculate_elu<APPROXIMATION_MODE, is_fp32_dest_acc_en, ITERATIONS>(slope);
    }
    static inline __attribute__((always_inline)) void init_op() { elu_init(); }
};

}  // namespace ckernel::sfpu
