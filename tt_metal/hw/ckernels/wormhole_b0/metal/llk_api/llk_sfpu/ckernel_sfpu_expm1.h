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
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];
        v = calculate_exponential_body_improved<APPROXIMATION_MODE>(v);
        dst_reg[0] = v - 1.0f;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void expm1_init() {
    exp_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel
