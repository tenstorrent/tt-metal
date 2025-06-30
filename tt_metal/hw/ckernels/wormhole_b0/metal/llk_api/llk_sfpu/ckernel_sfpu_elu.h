// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_elu(uint slope) {
    const bool FAST_APPROX = false;          // Elu does not use fast approximation.
    const bool SCALE_EN = false;             // Elu does not use scale.
    const bool SKIP_POSITIVE_CHECK = false;  // Elu does not skip positive check.
    // SFPU microcode
    vFloat s = Converter::as_float(slope);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];

        v_if(v < 0.0f) {
            vFloat v_exp = _calculate_exponential_body_improved_<
                APPROXIMATION_MODE,
                SCALE_EN,
                ITERATIONS,
                FAST_APPROX,
                SKIP_POSITIVE_CHECK>(v);
            v = s * (v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {
    const bool FAST_APPROX = false;  // Elu does not use fast approximation.
    exp_init<APPROXIMATION_MODE, FAST_APPROX>();
}

}  // namespace sfpu
}  // namespace ckernel
