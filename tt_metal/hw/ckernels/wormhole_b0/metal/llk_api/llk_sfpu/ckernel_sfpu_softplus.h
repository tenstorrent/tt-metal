// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_log.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(vFloat beta, vFloat beta_reciprocal, vFloat threshold) {
    vFloat a = dst_reg[0];
    vFloat a_beta = a * beta;
    v_if(a_beta < threshold) {
        exp_init<APPROXIMATION_MODE, false>();
        a = calculate_exponential_body<APPROXIMATION_MODE>(a_beta) + 1.0f;

        log_init<APPROXIMATION_MODE>();
        dst_reg[0] = a;
        calculate_log_body<false>(0);
        a = beta_reciprocal * dst_reg[0];
    }
    v_endif;
    dst_reg[0] = a;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    vFloat beta = Converter::to_float(param0);
    vFloat beta_reciprocal = Converter::to_float(param1);
    vFloat threshold = Converter::to_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {}

}  // namespace sfpu
}  // namespace ckernel
