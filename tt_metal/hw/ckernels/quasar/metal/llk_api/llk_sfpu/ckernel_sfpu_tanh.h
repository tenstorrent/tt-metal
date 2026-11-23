// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_polyval.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

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

// Blackhole's accurate fp32 tanh (gelu_tanh's fp32 Dest path), copied verbatim. It reads the constants
// _init_tanh_fp32_accurate_ programs, not the polynomial coefficients tanh_init loads.
// tanh(x): t = 0.5*expm1(abs(2*x)); sgn(x) * t / (t + 1)
sfpi_inline sfpi::vFloat _sfpu_tanh_fp32_accurate_(sfpi::vFloat x) {
    sfpi::vFloat a, r, s, f, w, y, scale, bias0;
    sfpi::vFloat j, t, rcp, x0, x1, y0;
    sfpi::vInt i, e, x_exp;
    sfpi::vMag m;

    // Calculate j = x * (2 * log2(e)), interleaved with a = abs(2*x), and i = round(abs(j)), clamped to [0, 255].

    j = x * sfpi::vConstFloatPrgm0;  // j = x * 2 * log2(e)
    a = x + x;
    // i = round(abs(j)), clamped to [0, 255].
    m = sfpi::convert<sfpi::vUInt8>(j, sfpi::RoundMode::Nearest);
    i = m;
    j = sfpi::convert<sfpi::vFloat>(m, sfpi::RoundMode::Nearest);

    a = sfpi::setsgn(a, 0);
    f = j * sfpi::vConstFloatPrgm1 + a;  // f = a - j * ln(2)

    // expm1(f)
    r = 1.974105835e-04f;
    r = r * f + 1.393318176e-3f;
    r = r * f + 8.331298828e-3f;
    r = r * f + 4.166680202e-2f;
    s = f * f;  // hide SFPMAD latency
    r = r * f + sfpi::vConstFloatPrgm2;
    w = 0.5f;
    r = __builtin_rvtt_sfpmad(r.get(), f.get(), w.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);

    e = i + 126;
    r = r * s + f;
    scale = sfpi::setexp(sfpi::vFloat(0.0f), e);
    bias0 = scale - w;

    // If a=±inf, converts to a finite value, otherwise if a=±NaN, converts to ±inf or ±NaN.
    // This gives y = <finite value> * 0.0 + 1.0 = 1.0 for non-NaN x, otherwise y = NaN.
    a = sfpi::as<sfpi::vFloat>(sfpi::as<sfpi::vInt>(a) - 1);
    x0 = r * scale + bias0;
    y = a * 0.0f + 1.0f;
    x1 = x0 + 1.0f;

    // `i` is round(abs(2*x/log(2))). For i >= 61, |x| is about 21 or larger,
    // so x0/(x0 + 1) is far within 0.5 ulp of 1.0f. Keep the preinitialized
    // saturated result; below that, refine the reciprocal estimate.
    v_if(i < 61) {
        // computes x0/x1 via reciprocal and residual correction
        rcp = sfpi::approx_recip(x1);
        t = -x1 * rcp + 1.0f;
        y = x;
        rcp = rcp * t + rcp;
        y0 = x0 * rcp;
        x_exp = sfpi::exexp(x, sfpi::ExponentMode::Biased);
        t = -x1 * y0 + x0;

        // For tiny inputs, tanh(x) rounds to x in fp32. `x_exp` is biased, so
        // 115 is 127 - 12; keep y=x for |x| < 2^-12 and use the corrected
        // ratio otherwise.
        v_if(x_exp >= 115) { y = t * rcp + y0; }
        v_endif;
    }
    v_endif;

    return sfpi::copysgn(y, x);
}

// Programs the constants _sfpu_tanh_fp32_accurate_ reads: Blackhole's tanh_init<false, true>.
inline void _init_tanh_fp32_accurate_() {
    sfpi::vConstFloatPrgm0 = 2.0f * 1.442695f;      // 2 * log2(e) == 2 / ln(2)
    sfpi::vConstFloatPrgm1 = -0.6931471805599453f;  // ln(2)
    sfpi::vConstFloatPrgm2 = 1.666667163e-1f;       // c1
}

// Blackhole's SFPI tanh polynomial and its init (softcap and situ_glu call them), copied from the
// Blackhole metal ckernel_sfpu_tanh.h; the SFPNONLINEAR kernel above stays Quasar's calculate_tanh.
// val * (0.999004364013671875 + val * (3.0897438526153564453125e-2 + val * (-0.4890659749507904052734375 + val *
// (0.281917631626129150390625 + val * (-6.6649019718170166015625e-2 + val *
// (5.876733921468257904052734375e-3))))));
constexpr float TANH_POLY_C1 = 0.999004364013671875f;
constexpr float TANH_POLY_C2 = 3.0897438526153564453125e-2f;
constexpr float TANH_POLY_C3 = -0.4890659749507904052734375f;

sfpi_inline sfpi::vFloat _sfpu_tanh_polynomial_(sfpi::vFloat x) {
    // For negative numbers, we compute tanh(-x) = -tanh(x)
    sfpi::vFloat val = sfpi::abs(x);  // set positive

    sfpi::vFloat result = PolynomialEvaluator::eval(
        val,
        0.0f,
        TANH_POLY_C1,
        TANH_POLY_C2,
        TANH_POLY_C3,
        sfpi::vConstFloatPrgm2,
        sfpi::vConstFloatPrgm1,
        sfpi::vConstFloatPrgm0);

    // For larger x, the polynomial approximation may exceed 1.0.
    // Since tanh(x) is bounded by [-1, 1], we clamp output to 1.0.
    result = sfpi::min(result, 1.0f);

    result = sfpi::copysgn(result, x);  // restore sign (i.e. tanh(-x) = -tanh(x))

    return result;
}

// Loads the coefficients _sfpu_tanh_polynomial_ reads (softcap, situ_glu) and the address mode
// calculate_tanh stores through. The template parameters are unused; they keep the Blackhole
// signature so tanh_tile_init needs no Quasar branch.
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en /*unused*/>
inline void tanh_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // calculate_tanh stores through ADDR_MOD_6: Dest += SFP_ROWS per pass.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
    sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
    sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
}

// Passes are pipelined (3 slots each): the next pass's load and the previous pass's store sit
// between an SFPNONLINEAR and its consumer. The store advances Dest through ADDR_MOD_6, set by
// tanh_init.
// Values carried between passes are pinned through l_reg[] so the recorded body ends in the state it
// starts in. Quasar cannot execute while recording, so the body is recorded, then replayed per pass.
inline constexpr sfpi::LRegs TANH_RESULT_LREG = sfpi::LRegs::LReg1;
inline constexpr unsigned TANH_REPLAY_START = 0;
inline constexpr unsigned TANH_REPLAY_LEN = 3;

template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_tanh() {
    static_assert(ITERATIONS >= 1, "calculate_tanh needs at least one pass");

    sfpi::l_reg[TANH_RESULT_LREG] = sfpi::approx_tanh(sfpi::dst_reg[0]);
    // Body for pass r: load pass r+1, store pass r (Dest += SFP_ROWS), tanh pass r+1.
    TTI_REPLAY(TANH_REPLAY_START, TANH_REPLAY_LEN, 0, 0, 0, 1);
    {
        sfpi::vFloat result = sfpi::l_reg[TANH_RESULT_LREG];
        sfpi::vFloat next = sfpi::dst_reg[1];
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = result;
        sfpi::l_reg[TANH_RESULT_LREG] = sfpi::approx_tanh(next);
    }
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        TTI_REPLAY(TANH_REPLAY_START, TANH_REPLAY_LEN, 0, 0, 0, 0);
    }
    sfpi::dst_reg[0].mode(ADDR_MOD_6) = sfpi::vFloat(sfpi::l_reg[TANH_RESULT_LREG]);
}

}  // namespace sfpu
}  // namespace ckernel
