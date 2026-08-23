// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_exp2.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

// erfc via two-branch decomposition:
//   x <= 0: direct degree-16/16 rational fit of erfc(x)        (range ~[1, 2])
//   x >  0: R(x) = erfc(x) * exp(+x^2) fit (smooth, O(0.11..1)),
//           reconstructed as exp2(-log2(e) * x^2) * R(x)
// Validated against scipy.special.erfc (fp64) over dense sweeps of [-5, 5]:
// max abs err 1.5e-3 -> see follow-up; refinement via internal refit pipeline
// targets fp32 few-ulp grade (cf. erf sibling n16/d16 fit).

constexpr std::array<float, 17> ERFC_NEG_NUM = {9.9999995544e-01f, -6.9538865569e-01f, -2.0563986567e-01f, 7.3229322727e-02f, 7.5069168380e-02f, 1.0836822558e-02f, -1.7522825296e-02f, -8.4350937600e-03f, 8.5945575997e-03f, 1.3059559288e-03f, -3.9319366337e-03f, 8.4335814198e-03f, 5.1398482070e-03f, -5.9652734210e-03f, -8.1382521307e-03f, -3.3542449152e-03f, -5.3556592644e-04f};
constexpr std::array<float, 17> ERFC_NEG_DEN = {1.0000000000e+00f, 4.3300027187e-01f, 2.8317023434e-01f, 1.8541296232e-02f, -5.8868011700e-02f, -3.2189246020e-02f, 3.2874805979e-03f, 1.6331766837e-02f, 6.9387143862e-03f, -9.7352819001e-03f, -1.1672490839e-02f, -3.4225433009e-04f, 1.2290744583e-03f, -3.2405243892e-03f, -4.1008078034e-03f, -1.6793849325e-03f, -2.6785448309e-04f};
constexpr std::array<float, 17> ERFC_POS_NUM = {1.0000009294e+00f, -4.8751487809e-01f, 1.4981833214e-01f, -1.9957305070e-03f, -2.4127814889e-02f, 4.3474122796e-03f, 9.0188314919e-03f, -3.2552444126e-03f, -4.4929121099e-03f, 2.9614775474e-03f, 7.1065942291e-04f, 4.5463479139e-04f, -3.0445464847e-03f, 1.6804537654e-03f, 2.5294996030e-03f, -3.2931177893e-03f, -6.6028407733e-08f};
constexpr std::array<float, 17> ERFC_POS_DEN = {1.0000000000e+00f, 6.4102441865e-01f, -1.2924824613e-01f, -2.2811226471e-02f, 2.0290363493e-02f, 5.0241531258e-03f, -7.4035619115e-03f, -2.4486001212e-03f, 4.4931087220e-03f, 1.0776566406e-03f, -3.6878205533e-03f, 1.2511137718e-03f, 1.3679107247e-03f, -1.6766851476e-03f, -2.7526834722e-04f, 4.5292674192e-03f, -5.8404238589e-03f};

template <bool APPROXIMATION_MODE>
inline void calculate_erfc() {
    constexpr int ITERATIONS = 8;
    constexpr float INV_LOG2E = -1.4426950408889634f;
    for (int di = 0; di < ITERATIONS; di++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat result;

        v_if(x <= 0.0f);
        {
            sfpi::vFloat numer = ERFC_NEG_NUM[16];
            sfpi::vFloat denom = ERFC_NEG_DEN[16];
            for (int k = 15; k >= 0; k--) {
                numer = numer * x + ERFC_NEG_NUM[k];
                denom = denom * x + ERFC_NEG_DEN[k];
            }
            result = numer / denom;
        }
        v_else;
        {
            sfpi::vFloat xx = x * x;
            sfpi::vFloat scale = _sfpu_exp2_fp32_accurate_(INV_LOG2E * xx);
            sfpi::vFloat numer = ERFC_POS_NUM[16];
            sfpi::vFloat denom = ERFC_POS_DEN[16];
            for (int k = 15; k >= 0; k--) {
                numer = numer * xx + ERFC_POS_NUM[k];
                denom = denom * xx + ERFC_POS_DEN[k];
            }
            result = scale * (numer / denom);
        }
        v_endif;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
