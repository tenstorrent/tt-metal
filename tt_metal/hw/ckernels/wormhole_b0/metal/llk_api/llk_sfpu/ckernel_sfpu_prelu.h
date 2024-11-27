// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_prelu(uint value) {
    // SFPU microcode
    Converter c_value;
    c_value.u = value;
    vFloat init = c_value.f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * init; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
