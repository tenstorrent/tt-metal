// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_polyval.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

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
// calculate_tanh stores through; SFPNONLINEAR's TANH_MODE itself uses its internal LUT. The template parameters are
// unused; they keep the Blackhole signature so tanh_tile_init needs no Quasar branch.
template <bool APPROXIMATION_MODE /*unused*/, bool is_fp32_dest_acc_en /*unused*/>
inline void tanh_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    // calculate_tanh stores through ADDR_MOD_6: Dest += SFP_ROWS per pass, the programming
    // init_typecast uses.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    sfpi::vConstFloatPrgm0 = 5.876733921468257904052734375e-3;
    sfpi::vConstFloatPrgm1 = -6.6649019718170166015625e-2;
    sfpi::vConstFloatPrgm2 = 0.281917631626129150390625;
}

// Pipelined across passes the way the Blackhole LUT kernels are (#65331): _calculate_tanh_sfp_rows_
// issues load / SFPNONLINEAR / store plus a TTINCRWC per pass of SFP_ROWS rows (4 slots), each
// waiting on the one before. Here the next pass's SFPLOAD and the previous pass's SFPSTORE sit
// between an SFPNONLINEAR and its consumer (3 slots per pass); the steady-state body is recorded
// once (executing as it records) and replayed; the store walks Dest through ADDR_MOD_6 (+SFP_ROWS, programmed by
// tanh_init the way init_typecast does), so no TTINCRWC per pass. Same instruction and mode as before.
template <int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_tanh() {
    static_assert(ITERATIONS >= 2, "calculate_tanh pipelines at least two passes");
    constexpr unsigned NEXT_PASS = ckernel::math::SFP_ROWS;  // Dest offset of the pass after the counter

    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, 0 /*pass 0*/);
    TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::TANH_MODE);
    // Body for pass r: load pass r+1, store pass r (Dest += SFP_ROWS), tanh pass r+1.
    load_replay_buf<0 /*start*/, 3 /*len*/, true /*exec_while_loading*/>([] {
        TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0 /*done*/, NEXT_PASS);
        TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
        TTI_SFPNONLINEAR(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpnonlinear::TANH_MODE);
    });
#pragma GCC unroll 8
    for (int d = 2; d < ITERATIONS; d++) {
        TTI_REPLAY(0 /*start*/, 3 /*len*/, 0 /*last*/, 0 /*set_mutex*/, 0 /*exec_while_loading*/, 0 /*load*/);
    }
    TTI_SFPSTORE(p_sfpu::LREG1, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_6, 0 /*done*/, 0);
}

}  // namespace sfpu
}  // namespace ckernel
