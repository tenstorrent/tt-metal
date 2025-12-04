// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline vFloat softplus(vFloat x) {
    /*
     Negative values, we use the exp21 function.
     Positive values, we use the polynomial approximation using remez minmax algorithm.
    */
    vFloat result = x;
    v_if(x < -5.0f) { result = _sfpu_exp_21f_(x); }
    v_elseif(x < 4.0f) {
        result = PolynomialEvaluator::eval(
            x,
            0.6924354434013367f,
            0.49275708198547363f,
            0.12142381817102432f,
            0.0031102809589356184f,
            -0.00330807245336473f,
            -0.00028794066747650504f,
            5.3185409342404455e-05f,
            7.1853546614875086e-06f,
            7.4961114648886e-08f);
    }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE>
inline void calculate_softplus_body(float beta, float beta_reciprocal, float threshold) {
    vFloat x = beta * dst_reg[0];
    v_if(x < threshold) { dst_reg[0] = beta_reciprocal * softplus(x); }
    v_endif;
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_softplus(uint param0, uint param1, uint param2) {
    float beta = Converter::as_float(param0);
    float beta_reciprocal = Converter::as_float(param1);
    float threshold = Converter::as_float(param2);
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_softplus_body<APPROXIMATION_MODE>(beta, beta_reciprocal, threshold);
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void softplus_init() {}

}  // namespace sfpu
}  // namespace ckernel
