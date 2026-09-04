// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_expm1_cw.h"
#include "llk_math_eltwise_sfpu_op.h"

namespace ckernel::sfpu {

// celu(x) = x for x>=0, alpha*(exp(x/alpha)-1) for x<0

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_celu(std::uint32_t param0, std::uint32_t param1) {
    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
// unroll 2: with expm1_cw_clamped inlined the loop body is large enough that
// partial unroll outperforms both full (unroll 8) and no-unroll (~0.8us on WH)
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result = alpha * expm1_cw_clamped(alpha_recip * x);

        v_if(x >= 0.0f) { result = x; }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// ---------------------------------------------------------------------------------------------------
// Celu<APPROX, DST_SYNC, DST_ACCUM, ITERATIONS>::calculate(dst_index, vector_mode, alpha, alpha_recip)
//   backs celu_tile / celu_tile_init (bare per-op init).
// ---------------------------------------------------------------------------------------------------
template <bool APPROXIMATION_MODE, DstSync DST_SYNC, bool DST_ACCUM, int ITERATIONS = 8>
struct Celu : SfpuUnaryOp<Celu<APPROXIMATION_MODE, DST_SYNC, DST_ACCUM, ITERATIONS>, DST_SYNC, DST_ACCUM> {
    static void kernel(std::uint32_t param0, std::uint32_t param1) {
        calculate_celu<APPROXIMATION_MODE, DST_ACCUM, ITERATIONS>(param0, param1);
    }
};
}  // namespace ckernel::sfpu
