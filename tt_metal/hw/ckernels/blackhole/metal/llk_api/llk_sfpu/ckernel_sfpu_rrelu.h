// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// RReLU eval/inference mode:
//   RReLU(x) = x              if x >= 0
//   RReLU(x) = slope * x      if x < 0
//   where slope = (lower + upper) / 2
//
// Parameters lower and upper are passed as bitcast uint32_t.
// The midpoint slope is computed on the SFPU.
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_rrelu(uint lower_u, uint upper_u) {
    // Reconstruct float parameters from bitcast uint32_t
    vFloat lower = Converter::as_float(lower_u);
    vFloat upper = Converter::as_float(upper_u);

    // Compute slope = (lower + upper) * 0.5
    vFloat slope = (lower + upper) * vFloat(0.5f);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat a = dst_reg[0];
        v_if(a < 0.0f) { a = a * slope; }
        v_endif;
        dst_reg[0] = a;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
