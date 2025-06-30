// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_exp.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_expm1() {
    const bool FAST_APPROX = false;          // Expm1 does not use fast approximation.
    const bool SCALE_EN = false;             // Expm1 does not use scale.
    const bool SKIP_POSITIVE_CHECK = false;  // Expm1 does not skip positive check.
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v = _calculate_exponential_body_improved_<
            APPROXIMATION_MODE,
            SCALE_EN,
            ITERATIONS,
            FAST_APPROX,
            SKIP_POSITIVE_CHECK>(v);
        dst_reg[0] = v - 1.0f;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void expm1_init() {
    const bool FAST_APPROX = false;  // Expm1 does not use fast approximation.
    exp_init<APPROXIMATION_MODE, FAST_APPROX>();
}

}  // namespace sfpu
}  // namespace ckernel
