// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_recip.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// erf(x) = x * P(x^2) / Q(x^2) on [-10, 10]. In x^2, P and Q have degree 3 and 4 for BF16 and 7 and 8
// for FP32.

#ifdef INP_FLOAT32
// P(t) = p0 + p1 t + ... + p7 t^7
constexpr int ERF_NUM_TERMS = 8;
constexpr float ERF_NUM[ERF_NUM_TERMS] = {
    1.1283791065e+00f,
    2.1477432549e-01f,
    6.2133435160e-02f,
    5.6230435148e-03f,
    6.1307044234e-04f,
    1.7678321456e-05f,
    2.7384647439e-08f,
    -2.8632063387e-10f};
// Q(t) = 1 + q1 t + ... + q8 t^8
constexpr int ERF_DEN_TERMS = 8;
constexpr float ERF_DEN[ERF_DEN_TERMS] = {
    5.2367275953e-01f,
    1.2961706519e-01f,
    1.9642570987e-02f,
    1.9545555115e-03f,
    1.3179056987e-04f,
    1.3156344494e-06f,
    -3.5153888689e-09f,
    -6.7350725691e-12f};
#else
// n8/d8 rational, coefficients aligned with WH v3 on-device refit (see PR #42540).
constexpr int ERF_NUM_TERMS = 4;
constexpr float ERF_NUM[ERF_NUM_TERMS] = {1.1280932447e+00f, 2.7609212279e-01f, 4.5400281738e-02f, 7.4481184425e-04f};
constexpr int ERF_DEN_TERMS = 4;
constexpr float ERF_DEN[ERF_DEN_TERMS] = {5.7439188334e-01f, 1.3675764810e-01f, 8.2844606784e-03f, 2.4813862145e-05f};
#endif

// Coefficient slots, highest power first (even: numerator, odd: denominator). The first stay in Prgm0..2
// (erf_init; accurate mode keeps Newton's 2.0 in Prgm0) and up to five LRegs, not two SFPLOADIs per pass.
template <bool APPROXIMATION_MODE>
constexpr int erf_prgm_slots() {
    return APPROXIMATION_MODE ? 3 : 2;
}

template <bool APPROXIMATION_MODE>
constexpr int erf_lreg_slots() {
    constexpr int SLOTS = ERF_NUM_TERMS + ERF_DEN_TERMS;
    return (SLOTS - erf_prgm_slots<APPROXIMATION_MODE>() <= 5) ? 5 : 4;
}

template <int SLOT>
constexpr float erf_slot_value() {
    if constexpr (SLOT % 2 == 0) {
        return ERF_NUM[ERF_NUM_TERMS - 1 - SLOT / 2];
    } else {
        return ERF_DEN[ERF_DEN_TERMS - 1 - SLOT / 2];
    }
}

struct ErfLRegs {
    sfpi::vFloat h0, h1, h2, h3, h4;
};

template <bool APPROXIMATION_MODE, int SLOT>
sfpi_inline sfpi::vFloat erf_coeff(const ErfLRegs& r) {
    constexpr int NUM_PRGM = erf_prgm_slots<APPROXIMATION_MODE>();
    constexpr int PRGM = APPROXIMATION_MODE ? SLOT : SLOT + 1;
    constexpr int H = SLOT - NUM_PRGM;
    if constexpr (SLOT < NUM_PRGM && PRGM == 0) {
        return sfpi::vConstFloatPrgm0;
    } else if constexpr (SLOT < NUM_PRGM && PRGM == 1) {
        return sfpi::vConstFloatPrgm1;
    } else if constexpr (SLOT < NUM_PRGM && PRGM == 2) {
        return sfpi::vConstFloatPrgm2;
    } else if constexpr (H >= erf_lreg_slots<APPROXIMATION_MODE>()) {
        return erf_slot_value<SLOT>();
    } else if constexpr (H == 0) {
        return r.h0;
    } else if constexpr (H == 1) {
        return r.h1;
    } else if constexpr (H == 2) {
        return r.h2;
    } else if constexpr (H == 3) {
        return r.h3;
    } else {
        return r.h4;
    }
}

template <bool APPROXIMATION_MODE, int H>
sfpi_inline sfpi::vFloat erf_lreg_value() {
    constexpr int SLOT = erf_prgm_slots<APPROXIMATION_MODE>() + H;
    if constexpr (H < erf_lreg_slots<APPROXIMATION_MODE>()) {
        return erf_slot_value<SLOT>();
    } else {
        return 0.0f;  // unused
    }
}

// Horner step I adds the next coefficient to both chains; Q has one more step, its constant 1.
// The two chains alternate so no SFPMAD waits on the one right before it.
template <bool APPROXIMATION_MODE, int I>
sfpi_inline void erf_horner(sfpi::vFloat& numer, sfpi::vFloat& denom, sfpi::vFloat x2, const ErfLRegs& r) {
    if constexpr (I < ERF_DEN_TERMS) {
        denom = denom * x2 + erf_coeff<APPROXIMATION_MODE, 2 * I + 1>(r);
    } else if constexpr (I == ERF_DEN_TERMS) {
        denom = denom * x2 + 1.0f;  // 1.0 is a constant register, no load
    }
    if constexpr (I < ERF_NUM_TERMS) {
        numer = numer * x2 + erf_coeff<APPROXIMATION_MODE, 2 * I>(r);
    }
    if constexpr (I < ERF_DEN_TERMS || I + 1 < ERF_NUM_TERMS) {
        erf_horner<APPROXIMATION_MODE, I + 1>(numer, denom, x2, r);
    }
}

// Whether BF16 DEST runs the generated erf kernel as one call over the whole tile.
inline constexpr bool erf_bf16_whole_tile = false;

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool is_fp32_dest_acc_en = true>
inline void calculate_erf() {
    static_assert(ERF_NUM_TERMS >= 2 && ERF_DEN_TERMS >= 2, "Horner chains start from two coefficients");
    const ErfLRegs r = {
        erf_lreg_value<APPROXIMATION_MODE, 0>(),
        erf_lreg_value<APPROXIMATION_MODE, 1>(),
        erf_lreg_value<APPROXIMATION_MODE, 2>(),
        erf_lreg_value<APPROXIMATION_MODE, 3>(),
        erf_lreg_value<APPROXIMATION_MODE, 4>()};
    // Unrolled so the coefficients above stay in LRegs; the compiler replays one pass.
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        // Clamping x^2 at 100 clamps |x| at 10, where erf is already +-1, and keeps the chains finite for inf.
        sfpi::vFloat x2 = sfpi::min(x * x, 100.0f);

        sfpi::vFloat denom = erf_coeff<APPROXIMATION_MODE, 1>(r) * x2 + erf_coeff<APPROXIMATION_MODE, 3>(r);
        sfpi::vFloat numer = erf_coeff<APPROXIMATION_MODE, 0>(r) * x2 + erf_coeff<APPROXIMATION_MODE, 2>(r);
        erf_horner<APPROXIMATION_MODE, 2>(numer, denom, x2, r);

        // x is read again from Dest instead of kept in an LReg, which frees one for a coefficient.
        numer = numer * sfpi::vFloat(sfpi::dst_reg[0]);

        // Q >= 1, so 1/Q needs no 0 / inf / NaN guard. Accurate mode adds two Newton steps.
        sfpi::vFloat y = sfpi::approx_recip(denom);
        if constexpr (!APPROXIMATION_MODE) {
            y = y * (sfpi::vConstFloatPrgm0 - denom * y);
            y = y * (sfpi::vConstFloatPrgm0 - denom * y);
        }
        sfpi::vFloat result = numer * y;
        // The fit overshoots 1 slightly in the tail and x * P keeps growing past |x| = 10. Clamp it, since a
        // value above 1 biases later ops (e.g. decomposed GELU in CLIP).
        result = sfpi::clamp(result, -1.0f, +1.0f);
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void erf_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    _init_reciprocal_<APPROXIMATION_MODE>();
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = erf_slot_value<0>();
        sfpi::vConstFloatPrgm1 = erf_slot_value<1>();
        sfpi::vConstFloatPrgm2 = erf_slot_value<2>();
    } else {
        sfpi::vConstFloatPrgm1 = erf_slot_value<0>();
        sfpi::vConstFloatPrgm2 = erf_slot_value<1>();
    }
}

}  // namespace ckernel::sfpu
