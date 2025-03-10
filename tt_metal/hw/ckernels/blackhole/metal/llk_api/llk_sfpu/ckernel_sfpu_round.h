// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    v_if(r < 0) {
        r = setsgn(r, 0);
        q = (q << 15) - r;
    }
    v_else { q = (q << 15) + r; }
    v_endif return q;
}

inline vFloat round_even(vFloat v) {
    vFloat result;
    v_if(sfpi::abs(v) < 0x1p30f) {
        result = int32_to_float(float_to_int31(v), 0);
        v_if(sfpi::abs(v - result) == 0.5F) {
            vInt res = float_to_int16(result);
            res = res & 0x7FFE;
            result = int32_to_float(res);
        }
        v_endif;
    }
    v_endif;
    return result;
}

inline constexpr float TABLE[] = {
    1e-45F, 1e-44F, 1e-43F, 1e-42F, 1e-41F, 1e-40F, 1e-39F, 1e-38F, 1e-37F, 1e-36F, 1e-35F, 1e-34F, 1e-33F, 1e-32F,
    1e-31F, 1e-30F, 1e-29F, 1e-28F, 1e-27F, 1e-26F, 1e-25F, 1e-24F, 1e-23F, 1e-22F, 1e-21F, 1e-20F, 1e-19F, 1e-18F,
    1e-17F, 1e-16F, 1e-15F, 1e-14F, 1e-13F, 1e-12F, 1e-11F, 1e-10F, 1e-9F,  1e-8F,  1e-7F,  1e-6F,  1e-5F,  1e-4F,
    1e-3F,  1e-2F,  1e-1F,  1e0F,   1e1F,   1e2F,   1e3F,   1e4F,   1e5F,   1e6F,   1e7F,   1e8F,   1e9F,   1e10F,
    1e11F,  1e12F,  1e13F,  1e14F,  1e15F,  1e16F,  1e17F,  1e18F,  1e19F,  1e20F,  1e21F,  1e22F,  1e23F,  1e24F,
    1e25F,  1e26F,  1e27F,  1e28F,  1e29F,  1e30F,  1e31F,  1e32F,  1e33F,  1e34F,  1e35F,  1e36F,  1e37F,  1e38F,
};

template <bool APPROXIMATE, int ITERATIONS = 8>
void calculate_round(const int decimals) {
    const auto exp10i = [](int n) {
        if (n > 38) {
            return 1.0F / 0.0F;
        }

        if (n < -45) {
            return 0.0F;
        }

        return TABLE[n + 45];
    };

    const vFloat coeff = exp10i(decimals);
    const vFloat inverse = exp10i(-decimals);

    for (int _ = 0; _ < ITERATIONS; ++_) {
        vFloat v = dst_reg[0];
        vFloat result = inverse * round_even(sfpi::abs(v) * coeff);
        result = setsgn(result, v);
        dst_reg[0] = result;
        ++dst_reg;
    }
}

}  // namespace ckernel::sfpu
