// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_trisc_common.h"
#include "cmath_common.h"

namespace ckernel {
namespace sfpu {

template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat val) {
    sfpi::vFloat result = sfpi::vConst0;

    constexpr float POLYNOMIAL_THRESHOLD = 0.6f;

    sfpi::vFloat abs_val = sfpi::abs(val);

    v_if(abs_val < POLYNOMIAL_THRESHOLD) {
        // Small |x|: Use minimax polynomial for better accuracy
        // Polynomial coefficients found with Sollya using the following command:
        // fpminimax(tanh(x)/x, [|0,2,4,6,8|], [|single...|], [-0.6; -2^(-40)] + [2^(-40); 0.6], relative);
        sfpi::vFloat x2 = val * val;

        sfpi::vFloat p = PolynomialEvaluator::eval(
            x2,
            0.999999940395355224609375f,
            -0.33332359790802001953125f,
            0.13310669362545013427734375f,
            -5.21197654306888580322265625e-2f,
            1.5497927553951740264892578125e-2f);

        result = val * p;
    }
    v_else {
        // Normal region: Use tanh(x) = sign(x) * (2*sigmoid(2*|x|) - 1)
        sfpi::vFloat two_x = 2.f * abs_val;
        sfpi::vFloat sig = _sfpu_sigmoid_<is_fp32_dest_acc_en>(two_x);

        // Compute 2*sigmoid(2*|x|) - 1
        sfpi::vFloat res_abs = 2.f * sig - sfpi::vConst1;
        result = sfpi::copysgn(res_abs, val);

        sfpi::vInt exponent = sfpi::exexp(val, sfpi::ExponentMode::NoDebias);
        // exp==255: NaN (default) or ±Inf (mantissa==0)
        v_if(exponent == 255) {
            sfpi::vInt mantissa = sfpi::exman(val);
            result = std::numeric_limits<float>::quiet_NaN();
            v_if(mantissa == 0) {
                sfpi::vFloat one = sfpi::vConst1;
                result = sfpi::copysgn(one, val);
            }
            v_endif;
        }
        v_endif;
    }
    v_endif;

    return result;
}

// Calculates tanh for number of rows of output SFPU ops (Quasar = 2 rows)
inline void _calculate_tanh_sfp_rows_() {
    TTI_SFPLOAD(
        p_sfpu::LREG0,
        p_sfpu::sfpmem::DEFAULT,
        ADDR_MOD_7,
        0,
        0);  // load from dest into lreg[0], uses ADDR_MOD_7 (set to all zeroes)
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::TANH_MODE);  // tanh via SFPU nonlinear unit
    TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 0);                           // store from lreg[1] into dest register
}

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_tanh() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        _calculate_tanh_sfp_rows_();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();  // does the dest_reg++ (increments by
                                                                                   // 2 rows)
    }
}

}  // namespace sfpu
}  // namespace ckernel
