// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpu/ckernel_sfpu_polyval.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// sech²(x) = 1/cosh²(x) polynomial approximation
// Remez minimax over [0, 4.0] with degree 12
// Avoids precision loss from 1-tanh²(x) computation which fails when tanh saturates
//
// Achieves Max ULP = 7 for |x| <= 4.0 (vs Max ULP = 15,139 with old 1-tanh² method)
// For |x| > 4.0, sech²(x) < 0.0005 and approaches BF16 precision limits
template <bool is_fp32_acc_to_dest_mode = true>
sfpi_inline vFloat _sfpu_sech2_polynomial_(vFloat x) {
    vFloat val = abs(x);  // sech²(-x) = sech²(x)

    // Polynomial coefficients (Remez minimax)
    // sech²(x) ≈ c0 + c1*x + c2*x² + ... + c12*x¹²
    vFloat result = PolynomialEvaluator::eval(
        val,
        1.000048079011522040e+00f,   // c0
        -3.914105753579200098e-03f,  // c1
        -9.478282605219310319e-01f,  // c2
        -2.626798850018942644e-01f,  // c3
        1.296857880675352792e+00f,   // c4
        -7.321966800441724876e-01f,  // c5
        -1.439733400234406713e-01f,  // c6
        3.675555549183296411e-01f,   // c7
        -2.059230945773536103e-01f,  // c8
        6.191285921380496049e-02f,   // c9
        vConstFloatPrgm2,            // c10 = -1.089204775237767597e-02
        vConstFloatPrgm1,            // c11 = 1.059831909816170981e-03
        vConstFloatPrgm0);           // c12 = -4.424893595600625488e-05

    // Clamp to [0, 1] - sech² is bounded by [0, 1]
    // For |x| > 4.0, polynomial may go negative, clamping handles this
    vFloat one = vConst1;
    vFloat zero = vConst0;
    vec_min_max(result, one);   // result = min(result, 1)
    vec_min_max(zero, result);  // result = max(0, result)

    if constexpr (!is_fp32_acc_to_dest_mode) {
        result = reinterpret<vFloat>(float_to_fp16b(result, 0));
    }

    return result;
}

template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH = 0, int ITERATIONS = 8>
inline void calculate_tanh_derivative() {
    if constexpr (APPROXIMATION_MODE) {
        // Fast approximation mode: use LUT-based tanh then 1-tanh²
        // This is less accurate but faster
        vUInt l0 = l_reg[LRegs::LReg0];
        vUInt l1 = l_reg[LRegs::LReg1];
        vUInt l2 = l_reg[LRegs::LReg2];

        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];

            if constexpr (!WITH_PRECOMPUTED_TANH) {
                val = lut(val, l0, l1, l2);
            }

            val = val * (-val) + vConst1;
            dst_reg[0] = val;

            dst_reg++;
        }

        l_reg[LRegs::LReg0] = l0;
        l_reg[LRegs::LReg1] = l1;
        l_reg[LRegs::LReg2] = l2;
    } else {
        // Accurate mode: use direct sech²(x) polynomial approximation
        // This avoids precision loss when tanh saturates to ±1
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat val = dst_reg[0];
            vFloat result = _sfpu_sech2_polynomial_<true>(val);
            dst_reg[0] = result;
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_init() {
    if constexpr (APPROXIMATION_MODE) {
        // LUT parameters for fast tanh approximation
        uint imm0 = 0x1DFF;  // 0.90625*x
        uint imm1 = 0x481A;  // 0.09375*x + 0.8125
        uint imm2 = 0xFF00;  // 1
        _sfpu_load_imm16_(0, imm0);
        _sfpu_load_imm16_(1, imm1);
        _sfpu_load_imm16_(2, imm2);
    } else {
        // Polynomial coefficients for accurate sech²(x) approximation
        // Last 3 coefficients stored in programmable registers
        vConstFloatPrgm0 = -4.424893595600625488e-05f;  // c12
        vConstFloatPrgm1 = 1.059831909816170981e-03f;   // c11
        vConstFloatPrgm2 = -1.089204775237767597e-02f;  // c10
    }
}

}  // namespace sfpu
}  // namespace ckernel
