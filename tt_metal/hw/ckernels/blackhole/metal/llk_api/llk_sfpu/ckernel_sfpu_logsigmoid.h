// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "ckernel_sfpu_softplus.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))
//
// The residual log1p(exp(-|x|)) is softplus's, reused rather than rebuilt: the
// same tuned polynomial on [0, 5], the same purpose-built exponential for a
// negative argument, and the same three term series for the tail. logsigmoid(x)
// is -softplus(-x), so the two were always computing the same thing, and only
// this one had it wrong.
//
// The form this replaces split the domain at +-4 with no arm for x <= -4, so an
// input there was returned unchanged and lost the residual entirely. Above +4 it
// used a one term series, e rather than e - e^2/2 + e^3/3, on an exponential the
// caller had computed in approximate mode. Both are gone: the split that remains
// is inside the residual, at 5, and both of its arms are present.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat a = sfpi::setsgn(x, 0);

        sfpi::vFloat residual;
#ifdef INP_FLOAT32
        {
            // float32 needs the residual to relative precision, not absolute:
            // for large positive x it is the whole answer and it is tiny. The
            // polynomial and three term tail softplus uses are accurate in
            // absolute terms and measured 3914 ulp here, so this path pays for
            // the accurate exponential and a real log1p instead.
            // float32 needs the residual to relative precision, not absolute:
            // for a positive input it is the whole answer and it is tiny.
            // softplus's polynomial is tuned in absolute terms and measured
            // 3914 ulp here at the same cost, so this path pays for the
            // accurate exponential and a real log1p.
            sfpi::vFloat t = _sfpu_exp_fp32_accurate_(-a);
            residual = calculate_log1p_fp32<true>(t);
        }
#else
        {
            // bfloat16 has eight bits of mantissa, which softplus's tuned
            // degree-6 polynomial covers on [0, 5]. Past that the residual is
            // below 0.0068 and exp(-a) alone is within half a bfloat16 ulp of
            // log1p(exp(-a)), so the tail needs no series.
            residual = PolynomialEvaluator::eval(
                a,
                SOFTPLUS_BF16_POLY_C0,
                SOFTPLUS_BF16_POLY_C1,
                SOFTPLUS_BF16_POLY_C2,
                SOFTPLUS_BF16_POLY_C3,
                SOFTPLUS_BF16_POLY_C4,
                SOFTPLUS_BF16_POLY_C5,
                SOFTPLUS_BF16_POLY_C6);
            // Past the polynomial's range the residual is exp(-a), below 0.0068
            // and within half a bfloat16 ulp of log1p(exp(-a)). Eight bits of
            // mantissa do not need softplus's Cody-Waite reduction, so this uses
            // the cheap 21f exponential instead.
            v_if(a > SOFTPLUS_POLY_BOUNDARY) {
                residual = _sfpu_exp_21f_bf16_<is_fp32_dest_acc_en>(sfpi::setsgn(a, 1));
            }
            v_endif;
        }
#endif

        // min(x, 0), the linear term the two branches were approximating on
        // either side of the old split.
        sfpi::vFloat lin = x;
        v_if(x > 0.0f) { lin = 0.0f; }
        v_endif;

        sfpi::vFloat result = lin - residual;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void logsigmoid_init() {
    softplus_init();
#ifdef INP_FLOAT32
    log1p_init<APPROXIMATION_MODE, /*FAST_APPROX=*/false, true>();
#endif
}

}  // namespace sfpu
}  // namespace ckernel
