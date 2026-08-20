// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log1p.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// logsigmoid(x) = min(x, 0) - log1p(exp(-|x|))
//
// The form this replaces split the domain at +-4 and had no arm for x <= -4, so
// an input there was returned unchanged, dropping the log1p(exp(x)) residual
// entirely. Above +4 it returned -exp(-x), a one term series that drops -e^2/2,
// and it took that exponential from an approximate exp computed by the caller.
//
// The identity has no split. The exponential argument is -|x|, so it is never
// positive and the result lies in (0, 1]: nothing can overflow, and log1p is
// evaluated exactly where it is accurate. min(x, 0) carries the linear term,
// which is what the +-4 branches were approximating on either side.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logsigmoid() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat t = _sfpu_exp_fp32_accurate_(-sfpi::setsgn(x, 0));
        sfpi::vFloat r = calculate_log1p_fp32<is_fp32_dest_acc_en>(t);

        sfpi::vFloat lin = x;
        v_if(x > 0.0f) { lin = 0.0f; }
        v_endif;

        sfpi::vFloat result = lin - r;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

// log1p reads its polynomial from the program constant registers and an SFPU
// helper called from another op's kernel does not inherit that op's init, so the
// coefficients are loaded here. The accurate exponential uses LREG[12..14] and
// does not collide with them.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void logsigmoid_init() {
    log1p_init<APPROXIMATION_MODE, /*FAST_APPROX=*/false, is_fp32_dest_acc_en>();
}

}  // namespace sfpu
}  // namespace ckernel
