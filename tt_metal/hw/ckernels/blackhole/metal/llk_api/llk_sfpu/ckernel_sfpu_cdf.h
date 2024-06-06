// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

#define POLYVAL5(coef4, coef3, coef2, coef1, coef0, val) \
    ((((coef4 * val + coef3) * val + coef2) * val + coef1) * val + coef0)

inline vFloat calculate_pos_cdf_appx(vFloat val) {
    //(0,2.5) interpolation polynomial coeffs  [ 0.0122792,  -0.05281024, -0.03048313,  0.41314081,  0.49866379]
    //(2.5,5) interpolation polynomial coeffs  [0.44656975,  0.58216001]

    // FIXME:
    // reuse LREG0-3 for storing coefficients and do product computation
    // const float coef_2dot5_to_5[4] = {-0.00221304f, -0.03253934f, -0.18027954f, -0.44656975f };
    // TTI_SFPLOADI(p_sfpu::LREG0, 0, 0xbb1108a6);
    // TTI_SFPLOADI(p_sfpu::LREG1, 0, 0xbd0547f9);
    // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbe389b33);
    // TTI_SFPLOADI(p_sfpu::LREG2, 0, 0xbee4a4ca);

    vFloat result;
    v_if(val < 2.5f) { result = POLYVAL5(0.0122792f, -0.05281024f, -0.03048313f, 0.41314081f, 0.49866379f, val); }
    v_else {
        // assume v >= 2.5f - 5
        // result = POLYVAL5(result,-0.00221304f,  0.03253934f, -0.18027954f,  0.44656975f,  0.58216001f, val);
        // result = ((vFloat)l_reg[LRegs::LReg0])*val + (vFloat)l_reg[LRegs::LReg1];
        // result = result*val + (vFloat)l_reg[LRegs::LReg2];
        // result = result*val + (vFloat)l_reg[LRegs::LReg3];
        result = 0.44656975f * val + 0.58216001f;
    }
    v_endif;

    v_if(result > 1.0f) { result = 1.0f; }
    v_endif;
    return result;
}

// compute the approximate value of CDF of normal distribution
inline vFloat calculate_cdf_appx(vFloat val, bool scaled = false) {
    vFloat result = 0.0f;
    vFloat val2 = 0.0;
    v_if(val < 0.0f) { val2 = -val; }
    v_else { val2 = val; }
    v_endif;

    result = calculate_pos_cdf_appx(val2);

    v_if(val < 0.0f) { result = 1.0f - result; }
    v_endif;

    if (scaled) {
        result *= val;  // scale
    }
    return result;
}

}  // namespace sfpu
}  // namespace ckernel
