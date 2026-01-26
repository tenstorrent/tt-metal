// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_quickgelu_appx.h"
// #include "sfpi/ckernel_sfpu_polyval.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_quickgelu() {
    // quickgelu(x) = x * sigmoid(1.702 * x)
    if constexpr (!APPROXIMATION_MODE) {
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            v_if(sfpi::dst_reg[0] < -3.0f) { x = 0.0f; }
            v_elseif(sfpi::dst_reg[0] < 0.0f) {
                x = x * POLYVAL5<sfpi::vFloat>(-0.000407f, 0.000125f, 0.047000f, 0.280537f, 0.502828f, 1.702f * x);
            }
            v_elseif(sfpi::dst_reg[0] < 3.0f) {
                x = x * (1.0f -
                         POLYVAL5<sfpi::vFloat>(-0.000407f, 0.000125f, 0.047000f, 0.280537f, 0.502828f, -1.702f * x));
            }
            v_endif;
            if constexpr (!is_fp32_dest_acc_en) {
                x = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(x, 0));
            }
            sfpi::dst_reg[0] = x;
            sfpi::dst_reg++;
        }
    } else {
        for (int d = 0; d < ITERATIONS; d++) {
            calculate_quickgelu_appx<ITERATIONS>();
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void quickgelu_init() {
    if constexpr (!APPROXIMATION_MODE) {
        _init_sfpu_reciprocal_<false>();
    } else {
        quickgelu_appx_init();
    }
}

}  // namespace sfpu
}  // namespace ckernel
