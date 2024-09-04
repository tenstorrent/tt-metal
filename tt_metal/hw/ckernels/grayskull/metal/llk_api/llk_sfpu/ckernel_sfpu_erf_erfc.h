// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL5(coef4,coef3,coef2,coef1,coef0,val) ( (((coef4*val + coef3)*val + coef2)*val + coef1)*val + coef0 )

template <bool APPROXIMATION_MODE>
sfpi_inline vFloat calculate_erf_body(vFloat x) {
    // assume x >= 0.
    vFloat result = 1.0f;
    v_if(x >= 3.0f) { result = 1.0f; }
    v_elseif(x >= 1.0f) { result = POLYVAL5(-0.03170029f, 0.31310241f, -1.1603072f, 1.91684792f, -0.19469693f, x); }
    v_elseif(x >= 0.0f) {
        result = POLYVAL5(0.166342190f, -0.476685015f, 0.0275416549, 1.12544048f, 0.0000661338118f, x);
    }
    v_else /* ( x <= 0.0f ) */ { result = 0.0f; }
    v_endif;
    // TODO: for higher accuracy (non APPROXIMATE) mode use higher degree polynomial.
    return result;
}

// TODO: Fix assertion error for accurate mode
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_erf() {
    for (int d = 0; d < ITERATIONS; d++) {
        // SFPU microcode:
        vFloat x = dst_reg[0];
        v_if(x < 0.0f) {
            x = -x;
            x = -calculate_erf_body<APPROXIMATION_MODE>(x);
        }
        v_else { x = calculate_erf_body<APPROXIMATION_MODE>(x); }
        v_endif;
        dst_reg[0] = x;
        dst_reg++;
    }
}

// TODO: Fix assertion error for accurate mode
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_erfc() {
// SFPU microcode:
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[0];
        v_if(x < 0.0f) { x = -x; x = 1.0 + (calculate_erf_body<APPROXIMATION_MODE>(x)); }
        v_else { x = 1.0 - (calculate_erf_body<APPROXIMATION_MODE>(x)); }
        v_endif;
        dst_reg[0] = x;
        dst_reg++;
    }
}

template <SfpuType operation, bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_sfpu_erf_erfc() {
    if constexpr (operation == SfpuType::erf) {
        calculate_erf<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::erfc) {
        calculate_erfc<APPROXIMATION_MODE, ITERATIONS>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
