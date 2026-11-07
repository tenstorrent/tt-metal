// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "lltt.h"
#include "ckernel_defs.h"
#include "cmath_common.h"  // math::reset_counters, p_setrwc

#include "ckernel_sfpu_exp.h"  // For _sfpu_round_to_nearest_int32_
#include "ckernel_sfpu_polyval.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_load_config.h"
#include "ckernel_sfpu_tanh.h"  // _sfpu_tanh_fp32_accurate_ for gelu_tanh
#include "sfpi.h"

namespace ckernel::sfpu {

// =============================================================================
// Fused x * exp(t) for GELU negative tail - avoids intermediate underflow
// =============================================================================
// Computes x * exp(t) where t = -x²/2 for x in the negative region (-13.375, -5].
// Input range: t ∈ [-89.5, -12.5] (from t = -x²/2 where x ∈ [-13.375, -5])
// Output range: x * exp(t) ∈ [~-4e-38, ~-1.4e-6]
//
// This function combines specialized inline exp() with a key insight to avoid
// intermediate underflow. Advantages over _sfpu_exp_f32_accurate_():
// 1. No overflow check (never happens for negative t)
// 2. No NaN check (input is computed from valid x)
// 3. Simpler FTZ handling via exponent field check
// 4. Uses degree-5 Taylor (vs degree-7 in accurate version)
// 5. ~15-18 ops vs ~25-30 for general-purpose exp
// 6. Fused multiply avoids intermediate underflow (see below)
//
// Method:
// 1. Range reduction: t = k·ln(2) + r, where k = round(t/ln2), |r| < 0.5
// 2. Polynomial: exp(r) via degree-5 Taylor series
// 3. Fused multiply: x * poly BEFORE 2^k scaling
// 4. Scaling: (x * poly) * 2^k via exponent bit manipulation (FREE)
//
// KEY INSIGHT: exp(t) alone may underflow (e.g., exp(-88.6) ≈ 3e-39 < BF16 min normal),
// but x * exp(t) may NOT underflow (e.g., -13.3 * 3e-39 = -4e-38 > BF16 min normal).
//
// Solution: Multiply x by poly BEFORE applying the 2^k exponent shift.
//   exp(t) = 2^k * poly(r)
//   x * exp(t) = x * 2^k * poly = (x * poly) * 2^k
//
// Since |x| ≈ 5-13 and poly ≈ 0.7-1.0, x * poly has a larger exponent than poly alone.
// When we then shift by 2^k, we're less likely to underflow.
//
// Example for x = -13.3125, t = -88.62:
//   OLD: poly=0.67, k=-128 → exp = 0.67 * 2^(-128) ≈ 2e-39 → FTZ to 0!
//   NEW: x*poly=-8.9, k=-128 → x*exp = -8.9 * 2^(-128) ≈ -2.6e-38 → representable!
//
// Performance: ~15-18 operations, no external function calls
// Accuracy: Max ULP ≤ 1 across entire (-13.375, -5] range
// =============================================================================
sfpi_inline sfpi::vFloat x_times_exp_negative_tail(sfpi::vFloat x, sfpi::vFloat t) {
    // Cody-Waite constants for extended precision range reduction
    constexpr float INV_LN2 = 1.4426950408889634f;  // 1/ln(2)
    constexpr float LN2_HI = -0.6931152343750000f;  // -ln(2) high bits (exact in float)
    constexpr float LN2_LO = -3.19461832987e-05f;   // -ln(2) low bits

    // Step 1: Range reduction - split t into k*ln(2) + r
    sfpi::vFloat z = t * INV_LN2;
    sfpi::vInt k_int;
    sfpi::vFloat k = _sfpu_round_to_nearest_int32_(z, k_int);

    // Step 2: Extended precision range reduction (Cody-Waite)
    sfpi::vFloat r = k * LN2_HI + t;
    r = k * LN2_LO + r;

    // Step 3: Polynomial approximation of exp(r) for |r| < 0.5
    constexpr float C2 = 0.5f;
    constexpr float C3 = 0.166666667f;
    constexpr float C4 = 0.0416666667f;
    constexpr float C5 = 0.00833333333f;
    sfpi::vFloat poly = PolynomialEvaluator::eval(r, 1.0f, 1.0f, C2, C3, C4, C5);

    // Step 4: FUSED MULTIPLY - multiply by x BEFORE exponent shift!
    // This is the key to avoiding intermediate underflow.
    // x * exp(t) = x * 2^k * poly = (x * poly) * 2^k
    sfpi::vFloat x_poly = x * poly;  // x * poly ≈ -9 to -13 (safe range)

    // Step 5: Exponent bit manipulation on the FUSED result
    sfpi::vInt xpoly_exp = sfpi::exexp(x_poly, sfpi::ExponentMode::Biased);  // Extract exponent of x*poly
    sfpi::vInt new_exp = xpoly_exp + k_int;                                  // Shift by 2^k

    // Step 6: FTZ check on FINAL result (x * exp(t)), not intermediate exp(t)
    sfpi::vFloat result = 0.0f;
    v_if(new_exp > 0) { result = sfpi::setexp(x_poly, new_exp); }
    v_endif;

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
void gelu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (APPROXIMATION_MODE) {
        sfpi::vConstFloatPrgm0 = 0.5f;
#ifndef DISABLE_SFPLOADMACRO
        TTI_SFPMAD(p_sfpu::LREG12, 0, p_sfpu::LREG7, 12, 0);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (0x80 | 0x40 | (2 << 3) | 4) << 8);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (0x40 | (4 << 3) | 3) << 8);
        TTI_SFPCONFIG(0, 4 + 0, 0);
        TTI_SFPCONFIG(0, 8, 1);
        TTI_SFPNOP;
        TTI_SFPNOP;
#endif

        // LUT segments (6-entry piecewise linear)
        sfpi::l_reg[sfpi::LRegs::LReg0] = sfpi::vLut16ss(0.1928f, 0.4939f);
        sfpi::l_reg[sfpi::LRegs::LReg4] = sfpi::vLut16ii(-0.00010443f, -0.1604f);

        sfpi::l_reg[sfpi::LRegs::LReg1] = sfpi::vLut16ss(0.6188f, 0.6099f);
        sfpi::l_reg[sfpi::LRegs::LReg5] = sfpi::vLut16ii(-0.2796f, -0.2635f);

        sfpi::l_reg[sfpi::LRegs::LReg2] = sfpi::vLut16ss(0.5402f, 0.5f);
        sfpi::l_reg[sfpi::LRegs::LReg6] = sfpi::vLut16ii(-0.1194f, 0.0f);
    } else if constexpr (is_fp32_dest_acc_en) {
        // FP32 accurate mode: rational erf evaluation requires reciprocal init
        sfpu_reciprocal_init<false>();
    } else {
        sfpi::vConstFloatPrgm0 = -0.72134752044f;
        sfpi::vConstFloatPrgm1 = 1.0017248f;
    }
}

#ifndef DISABLE_SFPLOADMACRO
template <int D, int N>
sfpi_inline void _gelu_appx_lm_rows_() {
    if constexpr (D < N) {
        TTI_SFPLOADMACRO((0 << 2) | 3, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 2 * D);
        TTI_SFPLUTFP32(p_sfpu::LREG7, 2);
        TTI_SFPNOP;
        _gelu_appx_lm_rows_<D + 1, N>();
    }
}
#endif

template <int ITERATIONS>
inline void calculate_gelu_appx() {
#ifndef DISABLE_SFPLOADMACRO
    TTI_SFPCONFIG(0xf00, 8, 1);
    TTI_SFPNOP;
    _gelu_appx_lm_rows_<0, ITERATIONS>();
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPNOP;
    TTI_SFPCONFIG(0, 8, 1);
    TTI_SFPNOP;
    TTI_SFPNOP;
#else
    sfpi::vLut16ss s01 = sfpi::l_reg[sfpi::LRegs::LReg0];
    sfpi::vLut16ss s23 = sfpi::l_reg[sfpi::LRegs::LReg1];
    sfpi::vLut16ss s45 = sfpi::l_reg[sfpi::LRegs::LReg2];
    sfpi::vLut16ii i01 = sfpi::l_reg[sfpi::LRegs::LReg4];
    sfpi::vLut16ii i23 = sfpi::l_reg[sfpi::LRegs::LReg5];
    sfpi::vLut16ii i45 = sfpi::l_reg[sfpi::LRegs::LReg6];

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in = sfpi::dst_reg[0];
        sfpi::vFloat half = sfpi::vConstFloatPrgm0;
        sfpi::vFloat half_in = in * half;
        sfpi::vFloat result = sfpi::lut(in, s01, i01, s23, i23, s45, i45, sfpi::LutSign::Update);
        result = half_in + result;

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }

    sfpi::l_reg[sfpi::LRegs::LReg0] = s01;
    sfpi::l_reg[sfpi::LRegs::LReg1] = s23;
    sfpi::l_reg[sfpi::LRegs::LReg2] = s45;
    sfpi::l_reg[sfpi::LRegs::LReg4] = i01;
    sfpi::l_reg[sfpi::LRegs::LReg5] = i23;
    sfpi::l_reg[sfpi::LRegs::LReg6] = i45;
#endif
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu() {
    if constexpr (APPROXIMATION_MODE) {
        calculate_gelu_appx<ITERATIONS>();
    } else if constexpr (is_fp32_dest_acc_en) {
        TTI_SFPLOADI(7, 2, 1267);
        TTI_SFPLOADI(7, 8, 16181);
        TTI_SFPLOADI(6, 2, 37728);
        TTI_SFPLOADI(6, 8, 45425);
        TTI_SFPLOADI(5, 2, 15226);
        TTI_SFPLOADI(5, 8, 13035);
        static_assert(ITERATIONS % 1 == 0, "gelu: 1 rows per loop");
        lltt::record<lltt::Exec>(0, 16);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPMUL(0, 7, 9, 1, 0);
        TTI_SFPLOADI(2, 0, 16672);
        TTI_SFPSWAP(0, 2, 1, 1);
        TTI_SFPNOP;
        TTI_SFPMUL(1, 1, 9, 3, 0);
        TTI_SFPLOADI(4, 2, 26636);
        TTI_SFPLOADI(4, 8, 44957);
        TTI_SFPLOADI(0, 2, 63524);
        TTI_SFPLOADI(0, 8, 44268);
        TTI_SFPMAD(0, 3, 6, 2, 0);
        TTI_SFPMAD(4, 3, 5, 0, 0);
        TTI_SFPLOADI(4, 2, 38107);
        TTI_SFPLOADI(4, 8, 13744);
        TTI_SFPMAD(2, 3, 4, 2, 0);
        TTI_SFPLOADI(4, 2, 19432);
        TTI_SFPLOADI(4, 8, 14228);
        TTI_SFPMAD(0, 3, 4, 4, 0);
        TTI_SFPLOADI(0, 2, 12611);
        TTI_SFPLOADI(0, 8, 14602);
        TTI_SFPMAD(2, 3, 0, 0, 0);
        TTI_SFPLOADI(2, 2, 46710);
        TTI_SFPLOADI(2, 8, 14880);
        TTI_SFPMAD(4, 3, 2, 4, 0);
        TTI_SFPLOADI(2, 2, 6144);
        TTI_SFPLOADI(2, 8, 15104);
        TTI_SFPMAD(0, 3, 2, 2, 0);
        TTI_SFPLOADI(0, 2, 16770);
        TTI_SFPLOADI(0, 8, 15288);
        TTI_SFPMAD(4, 3, 0, 0, 0);
        TTI_SFPLOADI(4, 2, 59765);
        TTI_SFPLOADI(4, 8, 15520);
        TTI_SFPMAD(2, 3, 4, 2, 0);
        TTI_SFPLOADI(4, 2, 32673);
        TTI_SFPLOADI(4, 8, 15742);
        TTI_SFPMAD(0, 3, 4, 4, 0);
        TTI_SFPLOADI(0, 2, 47702);
        TTI_SFPLOADI(0, 8, 15876);
        TTI_SFPMAD(2, 3, 0, 2, 0);
        TTI_SFPLOADI(0, 2, 60877);
        TTI_SFPLOADI(0, 8, 15963);
        TTI_SFPMAD(4, 3, 0, 0, 0);
        TTI_SFPLOADI(4, 2, 3947);
        TTI_SFPLOADI(4, 8, 16134);
        TTI_SFPMAD(2, 3, 4, 4, 0);
        TTI_SFPLOADI(2, 2, 28346);
        TTI_SFPLOADI(2, 8, 16272);
        TTI_SFPMAD(0, 3, 2, 0, 0);
        TTI_SFPMAD(4, 3, 10, 2, 0);
        TTI_SFPMUL(0, 1, 9, 4, 0);
        TTI_SFPMOV(0, 2, 3, 2);
        TTI_SFPSETMAN(0, 11, 3, 0);
        TTI_SFPMAD(12, 3, 13, 1, 0);
        TTI_SFPNOT(0, 2, 0, 0);
        TTI_SFPMAD(1, 3, 14, 2, 0);
        TTI_SFPSETMAN(0, 0, 1, 1);
        TTI_SFPMAD(3, 2, 10, 0, 0);
        TTI_SFPMULI(48896, 1, 0);
        TTI_SFPMAD(2, 0, 2, 0, 0);
        TTI_SFPLOAD(2, 0, 3, 0);
        TTI_SFPMAD(3, 0, 10, 3, 0);
        TTI_SFPNOP;
        TTI_SFPMAD(0, 3, 0, 0, 0);
        TTI_SFPLOADI(3, 0, 16128);
        TTI_SFPMUL(0, 1, 9, 1, 0);
        TTI_SFPLOADI(0, 2, 23791);
        TTI_SFPMUL(4, 1, 9, 4, 0);
        TTI_SFPLOADI(0, 8, 49329);
        TTI_SFPMAD(3, 4, 3, 1, 0);
        TTI_SFPMAD(2, 11, 0, 3, 0);
        TTI_SFPSWAP(0, 10, 1, 1);
        TTI_SFPNOP;
        TTI_SFPIADD(4095, 1, 4, 1);
        TTI_SFPLOADI(1, 0, 13056);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMUL(2, 1, 9, 0, 0);
        TTI_SFPMAD(2, 11, 2, 4, 0);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFPMOV(0, 0, 4, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSTORE(4, 0, 3, 0);
        TTI_INCRWC(0, 2, 0, 0);
#pragma GCC unroll 0
        for (int d = 1; d < ITERATIONS; d += 1) {
            lltt::replay(0, 16);
            TTI_SFPLOADI(4, 8, 14228);
            TTI_SFPMAD(0, 3, 4, 4, 0);
            TTI_SFPLOADI(0, 2, 12611);
            TTI_SFPLOADI(0, 8, 14602);
            TTI_SFPMAD(2, 3, 0, 0, 0);
            TTI_SFPLOADI(2, 2, 46710);
            TTI_SFPLOADI(2, 8, 14880);
            TTI_SFPMAD(4, 3, 2, 4, 0);
            TTI_SFPLOADI(2, 2, 6144);
            TTI_SFPLOADI(2, 8, 15104);
            TTI_SFPMAD(0, 3, 2, 2, 0);
            TTI_SFPLOADI(0, 2, 16770);
            TTI_SFPLOADI(0, 8, 15288);
            TTI_SFPMAD(4, 3, 0, 0, 0);
            TTI_SFPLOADI(4, 2, 59765);
            TTI_SFPLOADI(4, 8, 15520);
            TTI_SFPMAD(2, 3, 4, 2, 0);
            TTI_SFPLOADI(4, 2, 32673);
            TTI_SFPLOADI(4, 8, 15742);
            TTI_SFPMAD(0, 3, 4, 4, 0);
            TTI_SFPLOADI(0, 2, 47702);
            TTI_SFPLOADI(0, 8, 15876);
            TTI_SFPMAD(2, 3, 0, 2, 0);
            TTI_SFPLOADI(0, 2, 60877);
            TTI_SFPLOADI(0, 8, 15963);
            TTI_SFPMAD(4, 3, 0, 0, 0);
            TTI_SFPLOADI(4, 2, 3947);
            TTI_SFPLOADI(4, 8, 16134);
            TTI_SFPMAD(2, 3, 4, 4, 0);
            TTI_SFPLOADI(2, 2, 28346);
            TTI_SFPLOADI(2, 8, 16272);
            TTI_SFPMAD(0, 3, 2, 0, 0);
            TTI_SFPMAD(4, 3, 10, 2, 0);
            TTI_SFPMUL(0, 1, 9, 4, 0);
            TTI_SFPMOV(0, 2, 3, 2);
            TTI_SFPSETMAN(0, 11, 3, 0);
            TTI_SFPMAD(12, 3, 13, 1, 0);
            TTI_SFPNOT(0, 2, 0, 0);
            TTI_SFPMAD(1, 3, 14, 2, 0);
            TTI_SFPSETMAN(0, 0, 1, 1);
            TTI_SFPMAD(3, 2, 10, 0, 0);
            TTI_SFPMULI(48896, 1, 0);
            TTI_SFPMAD(2, 0, 2, 0, 0);
            TTI_SFPLOAD(2, 0, 3, 0);
            TTI_SFPMAD(3, 0, 10, 3, 0);
            TTI_SFPNOP;
            TTI_SFPMAD(0, 3, 0, 0, 0);
            TTI_SFPLOADI(3, 0, 16128);
            TTI_SFPMUL(0, 1, 9, 1, 0);
            TTI_SFPLOADI(0, 2, 23791);
            TTI_SFPMUL(4, 1, 9, 4, 0);
            TTI_SFPLOADI(0, 8, 49329);
            TTI_SFPMAD(3, 4, 3, 1, 0);
            TTI_SFPMAD(2, 11, 0, 3, 0);
            TTI_SFPSWAP(0, 10, 1, 1);
            TTI_SFPNOP;
            TTI_SFPIADD(4095, 1, 4, 1);
            TTI_SFPLOADI(1, 0, 13056);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPMUL(2, 1, 9, 0, 0);
            TTI_SFPMAD(2, 11, 2, 4, 0);
            TTI_SFPSETCC(0, 3, 0, 0);
            TTI_SFPMOV(0, 0, 4, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSTORE(4, 0, 3, 0);
            TTI_INCRWC(0, 2, 0, 0);
        }
    } else {
        TTI_SFPLOADI(7, 0, 49202);
        static_assert(ITERATIONS % 2 == 0, "gelu: 2 rows per loop");
        lltt::record<lltt::Exec>(0, 16);
        TTI_SFPLOAD(0, 0, 3, 0);
        TTI_SFPLOAD(1, 0, 3, 2);
        TTI_SFPMUL(0, 0, 9, 2, 0);
        TTI_SFPMUL(1, 1, 9, 3, 0);
        TTI_SFPLOADI(4, 2, 27209);
        TTI_SFPLOADI(5, 2, 27209);
        TTI_SFPLOADI(4, 8, 13227);
        TTI_SFPLOADI(5, 8, 13227);
        TTI_SFPLOADI(6, 2, 36986);
        TTI_SFPLOADI(6, 8, 46719);
        TTI_SFPMAD(2, 4, 6, 4, 0);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPLOADI(5, 2, 37522);
        TTI_SFPLOADI(5, 8, 14507);
        TTI_SFPMAD(2, 4, 5, 4, 0);
        TTI_SFPMAD(3, 6, 5, 6, 0);
        TTI_SFPLOADI(5, 2, 31486);
        TTI_SFPLOADI(5, 8, 47756);
        TTI_SFPMAD(2, 4, 5, 4, 0);
        TTI_SFPMAD(3, 6, 5, 5, 0);
        TTI_SFPLOADI(6, 2, 6851);
        TTI_SFPLOADI(6, 8, 15392);
        TTI_SFPMAD(2, 4, 6, 4, 0);
        TTI_SFPMAD(3, 5, 6, 5, 0);
        TTI_SFPLOADI(6, 2, 59468);
        TTI_SFPLOADI(6, 8, 48519);
        TTI_SFPMAD(2, 4, 6, 4, 0);
        TTI_SFPMAD(3, 5, 6, 6, 0);
        TTI_SFPLOADI(5, 2, 16938);
        TTI_SFPLOADI(5, 8, 16076);
        TTI_SFPMAD(2, 4, 5, 2, 0);
        TTI_SFPMAD(3, 6, 5, 4, 0);
        TTI_SFPLOADI(6, 0, 16128);
        TTI_SFPMAD(0, 2, 6, 3, 0);
        TTI_SFPMAD(1, 4, 6, 5, 0);
        TTI_SFPMUL(0, 3, 9, 2, 0);
        TTI_SFPMUL(1, 5, 9, 6, 0);
        TTI_SFPLOADI(4, 0, 16456);
        TTI_SFPMAD(0, 10, 4, 3, 0);
        TTI_SFPMAD(1, 10, 4, 5, 0);
        TTI_SFPMAD(0, 10, 7, 4, 0);
        TTI_SFPMUL(0, 0, 9, 0, 0);
        TTI_SFPSETCC(0, 3, 0, 4);
        TTI_SFPSETCC(0, 4, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 2, 2, 1);
        TTI_SFPSTORE(2, 0, 3, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPMAD(1, 10, 7, 3, 0);
        TTI_SFPMUL(1, 1, 9, 4, 0);
        TTI_SFPSETCC(0, 5, 0, 4);
        TTI_SFPSETCC(0, 3, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 6, 6, 1);
        TTI_SFPSTORE(6, 0, 3, 2);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPLOADI(2, 0, 17150);
        TTI_SFPMAD(0, 12, 2, 1, 0);
        TTI_SFPMAD(4, 12, 2, 5, 0);
        TTI_SFPEXEXP(0, 1, 3, 0);
        TTI_SFPEXEXP(0, 5, 6, 0);
        TTI_SFPEXMAN(0, 1, 0, 0);
        TTI_SFPEXMAN(0, 5, 4, 0);
        TTI_SFPSHFT(0, 3, 0, 0);
        TTI_SFPSHFT(0, 6, 4, 0);
        TTI_SFPEXEXP(0, 0, 2, 1);
        TTI_SFPEXEXP(0, 4, 1, 1);
        TTI_SFPEXMAN(0, 0, 5, 1);
        TTI_SFPEXMAN(0, 4, 3, 1);
        TTI_SFPCAST(5, 6, 0);
        TTI_SFPCAST(3, 0, 0);
        TTI_SFPLOADI(4, 2, 42008);
        TTI_SFPLOADI(5, 2, 42008);
        TTI_SFPLOADI(4, 8, 10156);
        TTI_SFPLOADI(5, 8, 10156);
        TTI_SFPLOADI(3, 2, 23258);
        TTI_SFPLOADI(3, 8, 13224);
        TTI_SFPMAD(6, 4, 3, 4, 0);
        TTI_SFPMAD(0, 5, 3, 5, 0);
        TTI_SFPMAD(6, 4, 13, 3, 0);
        TTI_SFPMAD(0, 5, 13, 4, 0);
        TTI_SFPSETEXP(0, 3, 2, 0);
        TTI_SFPSETEXP(0, 4, 1, 0);
        TTI_SFPLOAD(6, 0, 3, 0);
        TTI_SFPLOAD(0, 0, 3, 2);
        TTI_SFPLOADI(5, 2, 6529);
        TTI_SFPLOADI(3, 2, 6529);
        TTI_SFPLOADI(5, 8, 14919);
        TTI_SFPLOADI(3, 8, 14919);
        TTI_SFPLOADI(4, 2, 17221);
        TTI_SFPLOADI(4, 8, 15458);
        TTI_SFPMAD(6, 5, 4, 5, 0);
        TTI_SFPMAD(0, 3, 4, 4, 0);
        TTI_SFPLOADI(3, 2, 26648);
        TTI_SFPLOADI(3, 8, 15811);
        TTI_SFPMAD(6, 5, 3, 5, 0);
        TTI_SFPMAD(0, 4, 3, 4, 0);
        TTI_SFPLOADI(3, 2, 32398);
        TTI_SFPLOADI(3, 8, 16027);
        TTI_SFPMAD(6, 5, 3, 5, 0);
        TTI_SFPMAD(0, 4, 3, 3, 0);
        TTI_SFPLOADI(4, 0, 16064);
        TTI_SFPMAD(2, 5, 4, 5, 0);
        TTI_SFPMAD(1, 3, 4, 2, 0);
        TTI_SFPADDI(48832, 5, 0);
        TTI_SFPADDI(48832, 2, 0);
        TTI_SFPMUL(6, 5, 9, 1, 0);
        TTI_SFPMUL(0, 2, 9, 3, 0);
        TTI_SFPLOADI(4, 2, 23791);
        TTI_SFPLOADI(5, 2, 23791);
        TTI_SFPLOADI(4, 8, 49329);
        TTI_SFPLOADI(5, 8, 49329);
        TTI_SFPMAD(6, 11, 4, 2, 0);
        TTI_SFPMAD(0, 11, 5, 4, 0);
        TTI_SFPADDI(16456, 6, 0);
        TTI_SFPADDI(16456, 0, 0);
        TTI_SFPSETCC(0, 6, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 1, 1, 1);
        TTI_SFPSTORE(1, 0, 3, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 0, 0, 0);
        TTI_SFP_STOCH_RND(0, 0, 9, 3, 3, 1);
        TTI_SFPSTORE(3, 0, 3, 2);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 2, 0, 4);
        TTI_SFPSTORE(9, 0, 3, 0);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_SFPSETCC(0, 4, 0, 4);
        TTI_SFPSTORE(9, 0, 3, 2);
        TTI_SFPENCC(3, 0, 0, 10);
        TTI_INCRWC(0, 4, 0, 0);
#pragma GCC unroll 0
        for (int d = 2; d < ITERATIONS; d += 2) {
            lltt::replay(0, 16);
            TTI_SFPLOADI(5, 2, 31486);
            TTI_SFPLOADI(5, 8, 47756);
            TTI_SFPMAD(2, 4, 5, 4, 0);
            TTI_SFPMAD(3, 6, 5, 5, 0);
            TTI_SFPLOADI(6, 2, 6851);
            TTI_SFPLOADI(6, 8, 15392);
            TTI_SFPMAD(2, 4, 6, 4, 0);
            TTI_SFPMAD(3, 5, 6, 5, 0);
            TTI_SFPLOADI(6, 2, 59468);
            TTI_SFPLOADI(6, 8, 48519);
            TTI_SFPMAD(2, 4, 6, 4, 0);
            TTI_SFPMAD(3, 5, 6, 6, 0);
            TTI_SFPLOADI(5, 2, 16938);
            TTI_SFPLOADI(5, 8, 16076);
            TTI_SFPMAD(2, 4, 5, 2, 0);
            TTI_SFPMAD(3, 6, 5, 4, 0);
            TTI_SFPLOADI(6, 0, 16128);
            TTI_SFPMAD(0, 2, 6, 3, 0);
            TTI_SFPMAD(1, 4, 6, 5, 0);
            TTI_SFPMUL(0, 3, 9, 2, 0);
            TTI_SFPMUL(1, 5, 9, 6, 0);
            TTI_SFPLOADI(4, 0, 16456);
            TTI_SFPMAD(0, 10, 4, 3, 0);
            TTI_SFPMAD(1, 10, 4, 5, 0);
            TTI_SFPMAD(0, 10, 7, 4, 0);
            TTI_SFPMUL(0, 0, 9, 0, 0);
            TTI_SFPSETCC(0, 3, 0, 4);
            TTI_SFPSETCC(0, 4, 0, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 2, 2, 1);
            TTI_SFPSTORE(2, 0, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPMAD(1, 10, 7, 3, 0);
            TTI_SFPMUL(1, 1, 9, 4, 0);
            TTI_SFPSETCC(0, 5, 0, 4);
            TTI_SFPSETCC(0, 3, 0, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 6, 6, 1);
            TTI_SFPSTORE(6, 0, 3, 2);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPLOADI(2, 0, 17150);
            TTI_SFPMAD(0, 12, 2, 1, 0);
            TTI_SFPMAD(4, 12, 2, 5, 0);
            TTI_SFPEXEXP(0, 1, 3, 0);
            TTI_SFPEXEXP(0, 5, 6, 0);
            TTI_SFPEXMAN(0, 1, 0, 0);
            TTI_SFPEXMAN(0, 5, 4, 0);
            TTI_SFPSHFT(0, 3, 0, 0);
            TTI_SFPSHFT(0, 6, 4, 0);
            TTI_SFPEXEXP(0, 0, 2, 1);
            TTI_SFPEXEXP(0, 4, 1, 1);
            TTI_SFPEXMAN(0, 0, 5, 1);
            TTI_SFPEXMAN(0, 4, 3, 1);
            TTI_SFPCAST(5, 6, 0);
            TTI_SFPCAST(3, 0, 0);
            TTI_SFPLOADI(4, 2, 42008);
            TTI_SFPLOADI(5, 2, 42008);
            TTI_SFPLOADI(4, 8, 10156);
            TTI_SFPLOADI(5, 8, 10156);
            TTI_SFPLOADI(3, 2, 23258);
            TTI_SFPLOADI(3, 8, 13224);
            TTI_SFPMAD(6, 4, 3, 4, 0);
            TTI_SFPMAD(0, 5, 3, 5, 0);
            TTI_SFPMAD(6, 4, 13, 3, 0);
            TTI_SFPMAD(0, 5, 13, 4, 0);
            TTI_SFPSETEXP(0, 3, 2, 0);
            TTI_SFPSETEXP(0, 4, 1, 0);
            TTI_SFPLOAD(6, 0, 3, 0);
            TTI_SFPLOAD(0, 0, 3, 2);
            TTI_SFPLOADI(5, 2, 6529);
            TTI_SFPLOADI(3, 2, 6529);
            TTI_SFPLOADI(5, 8, 14919);
            TTI_SFPLOADI(3, 8, 14919);
            TTI_SFPLOADI(4, 2, 17221);
            TTI_SFPLOADI(4, 8, 15458);
            TTI_SFPMAD(6, 5, 4, 5, 0);
            TTI_SFPMAD(0, 3, 4, 4, 0);
            TTI_SFPLOADI(3, 2, 26648);
            TTI_SFPLOADI(3, 8, 15811);
            TTI_SFPMAD(6, 5, 3, 5, 0);
            TTI_SFPMAD(0, 4, 3, 4, 0);
            TTI_SFPLOADI(3, 2, 32398);
            TTI_SFPLOADI(3, 8, 16027);
            TTI_SFPMAD(6, 5, 3, 5, 0);
            TTI_SFPMAD(0, 4, 3, 3, 0);
            TTI_SFPLOADI(4, 0, 16064);
            TTI_SFPMAD(2, 5, 4, 5, 0);
            TTI_SFPMAD(1, 3, 4, 2, 0);
            TTI_SFPADDI(48832, 5, 0);
            TTI_SFPADDI(48832, 2, 0);
            TTI_SFPMUL(6, 5, 9, 1, 0);
            TTI_SFPMUL(0, 2, 9, 3, 0);
            TTI_SFPLOADI(4, 2, 23791);
            TTI_SFPLOADI(5, 2, 23791);
            TTI_SFPLOADI(4, 8, 49329);
            TTI_SFPLOADI(5, 8, 49329);
            TTI_SFPMAD(6, 11, 4, 2, 0);
            TTI_SFPMAD(0, 11, 5, 4, 0);
            TTI_SFPADDI(16456, 6, 0);
            TTI_SFPADDI(16456, 0, 0);
            TTI_SFPSETCC(0, 6, 0, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 1, 1, 1);
            TTI_SFPSTORE(1, 0, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSETCC(0, 0, 0, 0);
            TTI_SFP_STOCH_RND(0, 0, 9, 3, 3, 1);
            TTI_SFPSTORE(3, 0, 3, 2);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSETCC(0, 2, 0, 4);
            TTI_SFPSTORE(9, 0, 3, 0);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_SFPSETCC(0, 4, 0, 4);
            TTI_SFPSTORE(9, 0, 3, 2);
            TTI_SFPENCC(3, 0, 0, 10);
            TTI_INCRWC(0, 4, 0, 0);
        }
    }
}

// =============================================================================
// GELU tanh approximation in FP32:
//   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// =============================================================================

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_tanh() {
    constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;
    constexpr float GELU_TANH_K = 0.044715f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat p = GELU_TANH_K * x2 + 1.0f;
        sfpi::vFloat q = x * p;
        sfpi::vFloat u = SQRT_2_OVER_PI * q;
        sfpi::vFloat t = _sfpu_tanh_fp32_accurate_(u);

        // reload due to register pressure
        x = sfpi::dst_reg[0];
        sfpi::vFloat result = sfpi::copysgn(sfpi::vFloat(0.0f), x);

        sfpi::vFloat half_x = 0.5f * x;
        result = half_x * t + half_x;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

inline void gelu_tanh_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // initialise constants for _sfpu_tanh_fp32_accurate_
    tanh_init<false, true>();
}

// =============================================================================
// Fast GELU tanh approximation (opt-in, BF16-grade):
//   0.5 * x * (1 + tanh(u)) == x * sigmoid(2u) == x / (1 + exp(-2u)),  u = sqrt(2/pi) * (x + 0.044715 * x^3)
// exp(-2u) = 2^(-2u * log2(e)) uses the exp_21f bit trick (_float_to_int32_for_exp_21f_) with a degree-4
// polynomial for 2^f (max relative error 2.6e-6; exp_21f's quadratic is 1.7e-3), then two Newton steps on
// the reciprocal. Max error vs torch gelu(approximate="tanh") is ~1e-6 absolute, <= 1 BF16 ULP, in about
// 22 SFPU ops against ~38 for calculate_gelu_tanh. For FP32 outputs, prefer calculate_gelu_tanh.
// =============================================================================

// 2^f on [0, 1), fitted for relative error (mean +1.5e-8: unbiased), with the 2^-23 mantissa scale folded in:
// c_k * 2^(-23k). Blackhole uses a cheaper BF16-exact fit kept in programmable constants; here the reciprocal
// needs those constants, so the coefficients load inline.
// setexp() below overwrites the exponent, so the polynomial must stay in [1, 2) for every mantissa: it spans
// [1.0000025, 1.9999948] over all 2^23 (a degree-3 fit dips below 1 at f = 0 and is off by 2x there).
constexpr float GELU_TANH_FAST_EXP2_C0 = 1.0000025517e+00f;
constexpr float GELU_TANH_FAST_EXP2_C1 = 8.2612558619e-08f;
constexpr float GELU_TANH_FAST_EXP2_C2 = 3.4310902370e-15f;
constexpr float GELU_TANH_FAST_EXP2_C3 = 8.8111597259e-23f;
constexpr float GELU_TANH_FAST_EXP2_C4 = 2.7332817452e-30f;

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_tanh_fast() {
    constexpr float GELU_TANH_K = 0.044715f;
    constexpr float NEG_2_SQRT_2_OVER_PI_LOG2E = -2.3022081981f;  // -2 * sqrt(2/pi) * log2(e)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        sfpi::vFloat x2 = x * x;
        sfpi::vFloat p = GELU_TANH_K * x2 + 1.0f;
        sfpi::vFloat q = x * p;

        // Biased log2 of exp(-2u). Clamped so the exponent stays finite: 0 gives exp(-2u) ~ 0 (result x),
        // 254 gives exp(-2u) ~ 2^127, whose reciprocal flushes to 0 (result 0).
        sfpi::vFloat xlog2 = q * NEG_2_SQRT_2_OVER_PI_LOG2E + 127.0f;
        xlog2 = sfpi::clamp(xlog2, 0.0f, 254.0f);

        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vMag fractional_part = sfpi::exman(z);
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(
            frac,
            GELU_TANH_FAST_EXP2_C0,
            GELU_TANH_FAST_EXP2_C1,
            GELU_TANH_FAST_EXP2_C2,
            GELU_TANH_FAST_EXP2_C3,
            GELU_TANH_FAST_EXP2_C4);
        sfpi::vFloat e = sfpi::setexp(frac, exponential_part);

        // Two Newton steps: one alone lands at or below 1/d and biases |GELU| low.
        sfpi::vFloat result = x * sfpu_reciprocal_iter<2>(e + 1.0f);

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

inline void gelu_tanh_fast_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sfpu_reciprocal_init<false>();
}

// =============================================================================
// GELU Derivative - Polynomial Approximation
// =============================================================================
// GELU'(x) = Φ(x) + x*φ(x) where Φ is CDF, φ is PDF of standard normal
//
// Uses the identity GELU'(x) + GELU'(-x) = 1 (provable from Φ(x) + Φ(-x) = 1
// and φ(x) = φ(-x)), so GELU'(x) - 0.5 is an odd function of x, expressible as
// x * h(x²) for some function h. This halves the polynomial degree.
//
// Three active regions (plus zero default):
//   x >= 3.1719:          Saturation to 1
//   -3 <= x < 3.1719:     Core: 0.5 + x * h(x²), degree-8 in u=x²
//   -13.375 < x < -3:     Asymptotic: x * exp(-x²/2) with Mills ratio correction
//   x <= -13.375:          Zero (BF16 natural saturation)
// =============================================================================

// Degree-8 polynomial h(u) for GELU'(x) = 0.5 + x * h(x²) over [-3, 3.1719]
// Exploits odd-function decomposition: only 9 coefficients instead of 17.
// Evaluation: 1 MUL (u=x²) + 9 MAD (Horner on u) + 1 MUL (x*h) + 1 ADD (+0.5)
// = ~12 ops, vs ~32 ops for degree-16 in x.
// Coefficients from Sollya fpminimax.
constexpr float GELU_DERIV_H0 = 7.9788309336e-01f;
constexpr float GELU_DERIV_H1 = -2.6593554020e-01f;
constexpr float GELU_DERIV_H2 = 5.9766173363e-02f;
constexpr float GELU_DERIV_H3 = -9.4142099842e-03f;
constexpr float GELU_DERIV_H4 = 1.1061388068e-03f;
constexpr float GELU_DERIV_H5 = -9.7448595625e-05f;
constexpr float GELU_DERIV_H6 = 6.0921474869e-06f;
constexpr float GELU_DERIV_H7 = -2.3764030743e-07f;
constexpr float GELU_DERIV_H8 = 4.2734988881e-09f;

// GELU Derivative Evaluation with Polynomial Approximation
// Saturation thresholds derived from exhaustive BF16 research (DAZ+FTZ model):
// - Zero saturation: x <= -13.375 (GELU'(x) becomes 0 in BF16)
// - One saturation: x >= 3.1719 (GELU'(x) becomes exactly 1 in BF16)
// Note: GELU'(x) has a "hump" exceeding 1.0 for x in [0.77, 3.16]
// NaN: an fp32 destination returns x (NaN in, NaN out); a bf16 destination still gives
// 1.0 / 0 by the NaN's sign, since convert<vFloat16b> could not carry a NaN anyway.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat calculate_gelu_derivative_simple(sfpi::vFloat x) {
    sfpi::vFloat result = 0.0f;  // Default: 0 for x <= -13.375

    // fp32 destination: x*x is computed once, on every lane, because the NaN case after the
    // region chain needs it; the polynomial and tail regions reuse it. A bfloat16 destination
    // never reads x2_all and keeps computing x*x inside each region, as before.
    sfpi::vFloat x2_all;
    if constexpr (is_fp32_dest_acc_en) {
        x2_all = x * x;
    }

    // For x >= 3.1719, output saturates to 1 (verified saturation threshold)
    v_if(x >= 3.1719f) { result = 1.0f; }
    // Core region [-3, 3.1719]: GELU'(x) = 0.5 + x * h(x²)
    // Odd-function decomposition: GELU'(x) + GELU'(-x) = 1, so GELU'(x) - 0.5
    // is odd and can be written as x * h(x²). Degree-8 in u=x² (~12 ops vs ~32).
    v_elseif(x >= -3.0f) {
        sfpi::vFloat u;
        if constexpr (is_fp32_dest_acc_en) {
            u = x2_all;
        } else {
            u = x * x;
        }
        sfpi::vFloat h = PolynomialEvaluator::eval(
            u,
            GELU_DERIV_H0,
            GELU_DERIV_H1,
            GELU_DERIV_H2,
            GELU_DERIV_H3,
            GELU_DERIV_H4,
            GELU_DERIV_H5,
            GELU_DERIV_H6,
            GELU_DERIV_H7,
            GELU_DERIV_H8);
        result = 0.5f + x * h;
    }
    // Tail region (-13.375, -3): asymptotic formula with fused x*exp(t)
    // LEFT polynomial eliminated — the asymptotic formula achieves Max ULP ≤ 1
    // across the entire (-13.375, -3) range.
    // GELU'(x) ≈ φ(x) * (x - 1/x + 1/x³) where φ(x) = exp(-x²/2) / sqrt(2π)
    //
    // Uses x_times_exp_negative_tail() for fused x * exp(t) to avoid intermediate
    // underflow. Cody-Waite range reduction + degree-5 Taylor + direct exponent
    // bit manipulation.
    //
    // For APPROXIMATION_MODE=true: use simple x*φ(x) (~1% relative error)
    // For APPROXIMATION_MODE=false: use Mills ratio correction (<0.01% relative error)
    v_elseif(x > -13.375f) {
        constexpr float INV_SQRT_2PI = 0.3989422804014327f;  // 1/sqrt(2*pi)

        sfpi::vFloat x2;
        if constexpr (is_fp32_dest_acc_en) {
            x2 = x2_all;
        } else {
            x2 = x * x;
        }
        sfpi::vFloat t = x2 * (-0.5f);  // t = -x²/2

        sfpi::vFloat x_exp = x_times_exp_negative_tail(x, t);

        if constexpr (APPROXIMATION_MODE) {
            result = x_exp * INV_SQRT_2PI;
        } else {
            // 1 NR step suffices for BF16 (7 mantissa bits); FP32 needs 2 steps (23 bits).
            sfpi::vFloat inv_x2 = sfpu_reciprocal_iter<is_fp32_dest_acc_en ? 2 : 1>(x2);  // 1/x²
            sfpi::vFloat inv_x4 = inv_x2 * inv_x2;                                        // 1/x⁴
            sfpi::vFloat correction = 1.0f - inv_x2 + inv_x4;
            result = x_exp * INV_SQRT_2PI * correction;
        }
    }
    // For x <= -13.375, saturate to 0
    v_endif;

    // NaN (fp32 destination only; tenstorrent/tt-llk#1701 item 12). Every region test above is
    // an SFPMAD subtract followed by a sign test, and a NaN out of SFPMAD carries the product's
    // sign, so a NaN routes like an infinity of its own sign: +NaN saturated to 1.0 and -NaN
    // fell through to 0. The polynomial region is bounded, so no rewrite of those compares can
    // send both signs into it; test for NaN explicitly instead. x*x is +NaN for either NaN sign
    // and lies in [+0, +inf] for every other input, so its bits exceed +inf's exactly on NaN.
    // A bfloat16 destination is left alone: convert<vFloat16b> in the caller (SFPSTOCHRND)
    // turns any NaN into an infinity, so that arm could not return NaN anyway.
    if constexpr (is_fp32_dest_acc_en) {
        constexpr std::int32_t kInfBits = 0x7F800000;
        v_if(sfpi::as<sfpi::vInt>(x2_all) > kInfBits) { result = x; }
        v_endif;
    }

    return result;
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_gelu_derivative_polynomial() {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = calculate_gelu_derivative_simple<APPROXIMATION_MODE, is_fp32_dest_acc_en>(val);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void gelu_derivative_polynomial_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!APPROXIMATION_MODE) {
        // Call sfpu_reciprocal_init directly: gelu derivative uses sfpu_reciprocal_iter
        // inline (not _calculate_reciprocal_internal_), so SFPLOADMACRO fast-path init is
        // not needed. On BH, _init_reciprocal_ omits sfpu_reciprocal_init (it only
        // configures SFPLOADMACRO macros), so vConstFloatPrgm0=2.0f would be unset.
        //
        // Not keyed on is_fp32_dest_acc_en: the template arg to sfpu_reciprocal_iter<N>
        // selects the Newton-Raphson step count (2 for FP32, 1 for BF16), but
        // sfpu_reciprocal_init only seeds the initial estimate and does not vary
        // with N. A single <false> call therefore covers both precisions. The
        // asymmetry vs gelu_init() (which gained an fp32-specific branch) is
        // intentional — the derivative path has no LUT to load.
        sfpu_reciprocal_init<false>();
    }
}

}  // namespace ckernel::sfpu
