// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel::sfpu {

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

template <bool APPROXIMATE, int ITERATIONS = 8>
void calculate_round(const int decimals) {
    const auto powu = [](vFloat base, uint exp) {
        vFloat result = 1.0f;

        while (exp > 0) {
            if (exp & 1)
                result *= base;
            base *= base;
            exp >>= 1;
        }
        return result;
    };

    const auto exp10i = [powu](int exp) {
        if (exp < 0)
            return powu(0.1f, -exp);
        return powu(10.0f, exp);
    };

    const vFloat coeff = exp10i(decimals);
    const vFloat inverse = exp10i(-decimals);

    for (int _ = 0; _ < ITERATIONS; ++_) {
        vFloat v = dst_reg[0];
        dst_reg[0] = inverse * round_even(v * coeff);
        ++dst_reg;
    }
}

}  // namespace ckernel::sfpu
