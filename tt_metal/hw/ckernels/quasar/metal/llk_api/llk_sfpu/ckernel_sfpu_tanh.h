// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_polyval.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
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

// Blackhole's SFPI tanh polynomial and its init (softcap and situ_glu call them), copied from the
// Blackhole metal ckernel_sfpu_tanh.h; the SFPNONLINEAR kernel above stays Quasar's calculate_tanh.
// tanh(x): t = 0.5*expm1(abs(2*x)); sgn(x) * t / (t + 1)
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
    lltt::record(TANH_REPLAY_START, TANH_REPLAY_LEN);
    {
        sfpi::vFloat result = sfpi::l_reg[TANH_RESULT_LREG];
        sfpi::vFloat next = sfpi::dst_reg[1];
        sfpi::dst_reg[0].mode(ADDR_MOD_6) = result;
        sfpi::l_reg[TANH_RESULT_LREG] = sfpi::approx_tanh(next);
    }
#pragma GCC unroll 8
    for (int d = 1; d < ITERATIONS; d++) {
        lltt::replay(TANH_REPLAY_START, TANH_REPLAY_LEN);
    }
    sfpi::dst_reg[0].mode(ADDR_MOD_6) = sfpi::vFloat(sfpi::l_reg[TANH_RESULT_LREG]);
}

}  // namespace sfpu
}  // namespace ckernel
