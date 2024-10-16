// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline vInt float_to_int31(vFloat v) {
    vInt q = float_to_int16(v * 0x1p-15f, 0);
    vInt r = float_to_int16(v - int32_to_float(q, 0) * 0x1p15f, 0);
    return (q << 15) + r;
}

inline vFloat round_even(vFloat v) {
    v_if (sfpi::abs(v) < 0x1p30f) {
        v = int32_to_float(float_to_int31(v), 0);
    }
    v_endif;
    return v;
}

template <bool APPROX, int ITERATIONS = 8>
void calculate_round() {
    for (int _ = 0; _ < ITERATIONS; ++_) {
        *dst_reg = round_even(*dst_reg);
        ++dst_reg;
    }
}

}  // namespace sfpu
}  // namespace ckernel
