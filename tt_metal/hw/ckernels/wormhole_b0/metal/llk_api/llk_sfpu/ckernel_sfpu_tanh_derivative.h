// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp.h"
#include "sfpu/ckernel_sfpu_recip.h"

namespace ckernel {
namespace sfpu {

// Legacy tanh derivative using 1 - tanh²(x) via LUT.
// WARNING: This has catastrophic cancellation for |x| > ~3.4 (Max ULP = 15,140).
// Kept for backward compatibility. Use calculate_tanh_derivative_sech2 instead.
template <bool APPROXIMATION_MODE, int WITH_PRECOMPUTED_TANH = 0, int ITERATIONS = 8>
inline void calculate_tanh_derivative() {
    vUInt l0 = l_reg[LRegs::LReg0];
    vUInt l1 = l_reg[LRegs::LReg1];
    vUInt l2 = l_reg[LRegs::LReg2];

    // tanh'(x) = 1 - (tanh(x))^2
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
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_init() {
    uint imm0;
    uint imm1;
    uint imm2;
    imm0 = 0x1DFF;  // 0.90625*x
    imm1 = 0x481A;  // 0.09375*x + 0.8125
    imm2 = 0xFF00;  // 1
    _sfpu_load_imm16_(0, imm0);
    _sfpu_load_imm16_(1, imm1);
    _sfpu_load_imm16_(2, imm2);
}

// Accurate tanh derivative using sech²(x) = 4·exp(-2|x|) / (1 + exp(-2|x|))²
// This avoids the catastrophic cancellation in 1 - tanh²(x).
// Uses _sfpu_exp_f32_accurate_ (< 1 FP32 ULP) and _sfpu_reciprocal_<2> (Newton-Raphson).
//
// Three regions:
//   |x| < 10:      Full formula 4·exp(-2|x|) / (1+exp(-2|x|))² (denominator matters)
//   10 <= |x| < 45: Asymptotic exp(-2|x| + ln4) (folds ×4 into exp to avoid FP32 FTZ)
//   |x| >= 45:     Clamp to 0 (result < BF16 min normal)
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_tanh_derivative_sech2() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::vConst0;

        // sech²(x) is an even function: sech²(-x) = sech²(x)
        sfpi::vFloat a = sfpi::abs(val);

        v_if(a < 10.0f) {
            // Core region: full formula with reciprocal
            sfpi::vFloat t = a * (-2.0f);
            sfpi::vFloat e = _sfpu_exp_f32_accurate_(t);
            sfpi::vFloat denom = sfpi::vConst1 + e;
            sfpi::vFloat inv_denom = _sfpu_reciprocal_<2>(denom);
            result = 4.0f * e * inv_denom * inv_denom;
        }
        v_elseif(a < 45.0f) {
            // Asymptotic region: exp(-2|x|) << 1, so (1+exp(-2|x|))² ≈ 1.
            // Compute 4·exp(-2|x|) = exp(-2|x| + ln4) in one exp call.
            // This avoids FP32 FTZ: exp(-87.5) is FP32 denormal, but
            // exp(-87.5 + 1.386) = exp(-86.114) ≈ 3.9e-38 is normal.
            constexpr float LN4 = 1.3862943611198906f;
            sfpi::vFloat t = a * (-2.0f) + LN4;
            result = _sfpu_exp_f32_accurate_(t);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void tanh_derivative_sech2_init() {
    _init_reciprocal_<false, false>();
}

}  // namespace sfpu
}  // namespace ckernel
