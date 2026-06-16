// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_log.h"
#include "sfpu/ckernel_sfpu_recip.h"

#include "ckernel_sfpu_piecewise_rational.h"

namespace ckernel::sfpu {

// ======================================================================
// LUT-based digamma via piecewise rational P(x)/Q(x)
//
// BF16: n12/d11, 2 segment(s), range [0.01, 102.0]
// FP32: n10/d9, 2 segment(s), range [0.01, 102.0]
// ======================================================================

#ifdef INP_FLOAT32
constexpr uint32_t DIGAMMA_NUM_DEGREE = 10;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 9;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 2;
constexpr uint32_t DIGAMMA_LUT_SIZE = 45;
constexpr std::array<float, 45> DIGAMMA_LUT = {
    {1.0000000000e-02f,  5.1005000000e+01f, 1.0200000000e+02f, -4.2596286535e-01f, -1.2459139824e+00f,
     -7.0374596119e-01f, 3.5418295860e-01f, 3.9686101675e-01f, 1.0760356486e-01f,  1.1159370653e-02f,
     4.5299832709e-04f,  6.4513715188e-06f, 2.3348954770e-08f, 3.3629592167e-12f,  -6.3762914866e-08f,
     4.2596703768e-01f,  1.0000000000e+00f, 8.2736301422e-01f, 3.0081826448e-01f,  4.9985647202e-02f,
     3.7117889151e-03f,  1.1463041301e-04f, 1.2610012732e-06f, 3.3732798776e-09f,  1.3856337913e+05f,
     -2.7915017828e+05f, 5.5189096727e+04f, 2.6230285816e+04f, 2.0102400217e+03f,  5.4774037756e+01f,
     6.1410588531e-01f,  2.8495312547e-03f, 4.9248967852e-06f, 2.2963254970e-09f,  3.8530168912e-14f,
     -1.6724299601e+05f, 1.0000000000e+00f, 6.3677231206e+04f, 1.1013851737e+04f,  5.9383811201e+02f,
     1.2890647910e+01f,  1.2061770104e-01f, 4.7490052009e-04f, 6.9240316615e-07f,  2.5925175628e-10f}};

#else

constexpr uint32_t DIGAMMA_NUM_DEGREE = 12;
constexpr uint32_t DIGAMMA_DEN_DEGREE = 11;
constexpr uint32_t DIGAMMA_NUM_SEGMENTS = 2;
constexpr uint32_t DIGAMMA_LUT_SIZE = 53;
constexpr std::array<float, 53> DIGAMMA_LUT = {
    {1.0000000000e-02f,  5.1005000000e+01f,  1.0200000000e+02f,  1.1232896805e+01f,  5.4837932587e+00f,
     -5.3902623117e+01f, -3.7616708696e+01f, 1.3208077133e+01f,  1.5625772953e+01f,  4.2468094435e+00f,
     4.8425609153e-01f,  2.4947039550e-02f,  5.5918553608e-04f,  4.7726717618e-06f,  1.1066160266e-08f,
     1.0473417859e-12f,  -1.9764498267e-07f, -1.1232894897e+01f, 1.0000000000e+00f,  3.4848090112e+01f,
     3.2649136305e+01f,  1.1910055429e+01f,  2.0447196774e+00f,  1.7210583540e-01f,  6.9577881368e-03f,
     1.2539303970e-04f,  8.5822975660e-07f,  1.5135111497e-09f,  -1.1970819551e+08f, 2.5625265499e+08f,
     -2.8415626732e+07f, -4.5414422193e+07f, -6.5386237758e+06f, -3.4774655950e+05f, -8.4142866663e+03f,
     -9.9024325624e+01f, -5.6814976360e-01f, -1.5061196535e-03f, -1.6082146010e-06f, -4.9102502379e-10f,
     -5.4932387401e-15f, 8.8576622571e+07f,  1.0000000000e+00f,  -8.6676697066e+07f, -2.4838010772e+07f,
     -2.3597817534e+06f, -9.8211568465e+04f, -1.9824152142e+03f, -2.0056206722e+01f, -1.0026611704e-01f,
     -2.3192374371e-04f, -2.1315511702e-07f, -5.3136111083e-11f}};

#endif

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_digamma() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Piecewise-rational LUT, fit on [0.01, 102].
        sfpi::vFloat result =
            piecewise_rational_eval<DIGAMMA_NUM_DEGREE, DIGAMMA_DEN_DEGREE, DIGAMMA_NUM_SEGMENTS, DIGAMMA_LUT_SIZE>(
                DIGAMMA_LUT, x);

        // Large-x asymptotic (Bernoulli series). Above the LUT fit range the rational
        // extrapolates poorly (issue #45520: "behaves bad for x>1000"); the asymptotic
        // digamma(x) = ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ... is ~exact for large x
        // (truncation error ~8e-11 at x=102), restoring the (1, inf) support the pre-LUT
        // composite op provided. Crossover at the LUT's upper bound 102 is seamless.
        // NOTE: uses the register-free (bf16-grade) log body, so large-x fp32 is only
        // bf16-accurate. #45520 targets bf16 ULP; an fp32-accurate log here would collide
        // with the reciprocal's vConstFloatPrgm0, so fp32 large-x is intentionally left as-is.
        v_if(x > 102.0f) {
            sfpi::vFloat inv_x = _sfpu_reciprocal_<2>(x);
            sfpi::vFloat inv_x2 = inv_x * inv_x;
            // ln(x) - inv_x*0.5 - inv_x2*(1/12 - inv_x2/120)
            sfpi::vFloat bern = sfpi::vFloat(0.0833333333f) - inv_x2 * sfpi::vFloat(0.0083333333f);
            result = _calculate_log_body_no_init_(x) - inv_x * sfpi::vFloat(0.5f) - inv_x2 * bern;
            // digamma(+inf) = +inf; the log approximation clamps inf to a finite value, so
            // restore it explicitly (exp field all-ones, zero mantissa => infinity).
            v_if(sfpi::exexp(x) == 128 && sfpi::exman(x) == 0) { result = std::numeric_limits<float>::infinity(); }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void digamma_init() {
    sfpu_reciprocal_init();
}

}  // namespace ckernel::sfpu
