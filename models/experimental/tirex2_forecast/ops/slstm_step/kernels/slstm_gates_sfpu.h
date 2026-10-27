// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Whole sLSTM gate chain in one SFPU pass over DST (fp32), replacing ~20 per-op passes and their inits.
// Slots (tiles): 0 i, 1 f, 2 z, 3 o, 4 m, 5 c, 6 n on input; on output 3 y, 4 m', 5 c', 6 n' (0-2 scratch).
//   lfm = logsig(f) + m; m' = max(i, lfm); ig = exp(i - m'); fg = exp(lfm - m')
//   c' = fg c + ig tanh z; n' = max(fg n + ig, 1); y = sigmoid(o) c' / n'
// first (t = 0): m' = i, c' = tanh z, n' = 1, y = sigmoid(o) tanh z.
// Uses only literal constants, no programmable constant registers or LUTs.

#if defined(TRISC_MATH)

#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"

namespace ckernel::sfpu {

// Timing-only sub-ablation (results wrong): bit 1 exp -> x, bit 2 recip without Newton, bit 4 logsig poly -> f,
// bit 8 no logsig exp tail.
#ifndef SLSTM_GABL
#define SLSTM_GABL 0
#endif

#ifndef SLSTM_FASTEXP
#define SLSTM_FASTEXP 1
#endif

sfpi_inline sfpi::vFloat _slstm_exp_(sfpi::vFloat x) {
    if constexpr (SLSTM_GABL & 1) {
        return x;
    } else {
        // Unguarded exp: clamp to the normal range instead of the inf/underflow branches.
        v_if(x < -87.0f) { x = -87.0f; }
        v_endif;
        v_if(x > 87.0f) { x = 87.0f; }
        v_endif;
#if SLSTM_FASTEXP
        // One-constant Cody-Waite + degree-5 poly on [-ln2/2, ln2/2] (rel err ~3e-6).
        sfpi::vFloat j = 1.442695f * x;
        sfpi::vSMag16 sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
        j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
        sfpi::vFloat f = j * -0.693147182f + x;
#if SLSTM_FASTEXP == 2
        // Degree-4 poly (rel err ~6e-6), one fewer FMA per exp.
        sfpi::vFloat r = 4.15458507e-2f;
        r = r * f + 1.67766179e-1f;
        r = r * f + 5.00021374e-1f;
        sfpi::vInt i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
        r = r * f + 9.99972398e-1f;
#else
        sfpi::vFloat r = 8.37312452e-3f;
        r = r * f + 4.16695364e-2f;
        r = r * f + 1.66664720e-1f;
        r = r * f + 4.99999851e-1f;
        sfpi::vInt i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
        r = r * f + 1.0f;
#endif
        i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
        r = r * f + 1.0f;
        return sfpi::setexp(r, sfpi::exexp(r, sfpi::ExponentMode::Biased) + i);
#else
        return _sfpu_exp_fp32_accurate_<true>(x);
#endif
    }
}

// 1 / x with Newton refinement (x finite, nonzero, here x >= 1); SLSTM_NEWTON steps (default 1).
#ifndef SLSTM_NEWTON
#define SLSTM_NEWTON 1
#endif
sfpi_inline sfpi::vFloat _slstm_recip_(sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    if constexpr (!(SLSTM_GABL & 2)) {
        y = y * (2.0f - x * y);
        if constexpr (SLSTM_NEWTON > 1) {
            y = y * (2.0f - x * y);
        }
    }
    return y;
}

sfpi_inline sfpi::vFloat _slstm_sigmoid_(sfpi::vFloat x) {
    return _slstm_recip_(1.0f + _slstm_exp_(-x));
}

// log(sigmoid(f)) = min(f, 0) - log1p(exp(-|f|)); log1p(exp(-a)) from softplus' fp32 poly on [0, 5]
// and exp tail e (1 - e/2 + e^2/3) above.
sfpi_inline sfpi::vFloat _slstm_logsig_(sfpi::vFloat f) {
    if constexpr (SLSTM_GABL & 4) {
        return f;
    }
    sfpi::vFloat a = sfpi::setsgn(f, 0);
    sfpi::vFloat r = -4.8245715334e-07f;
    r = r * a + 2.1285692128e-05f;
    r = r * a + -3.4358495031e-04f;
    r = r * a + 2.7290175203e-03f;
    r = r * a + -1.0528374463e-02f;
    r = r * a + 5.6753782555e-03f;
    r = r * a + 1.2186349183e-01f;
    r = r * a + -4.9926245213e-01f;
    r = r * a + 6.9310557842e-01f;
    if constexpr (!(SLSTM_GABL & 8)) {
        v_if(a > 5.0f) {
            // e = exp(-a) < 6.7e-3 here, so a degree-3 exp (rel err ~6e-4) is ~4e-6 absolute; a <= 30 keeps setexp normal.
            sfpi::vFloat x = -a;
            v_if(x < -30.0f) { x = -30.0f; }
            v_endif;
            sfpi::vFloat j = 1.442695f * x;
            sfpi::vSMag16 sm = sfpi::convert<sfpi::vSMag16>(j, sfpi::RoundMode::Nearest);
            j = sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);
            sfpi::vFloat g = j * -0.693147182f + x;
            sfpi::vFloat e = 0.166666672f;
            e = e * g + 0.5f;
            e = e * g + 1.0f;
            sfpi::vInt i = sfpi::abs(sfpi::as<sfpi::vInt>(sm));
            e = e * g + 1.0f;
            i = sfpi::as<sfpi::vInt>(sfpi::copysgn(sfpi::as<sfpi::vFloat>(i), j));
            e = sfpi::setexp(e, sfpi::exexp(e, sfpi::ExponentMode::Biased) + i);
            r = e * (1.0f + e * -0.5f);
        }
        v_endif;
    }
    v_if(f < 0.0f) { r = f - r; }
    v_else { r = -r; }
    v_endif;
    return r;
}

template <bool first>
inline void calculate_slstm_gates() {
    constexpr uint32_t ts = 32;  // one DST tile in SFPU row addressing
#pragma GCC unroll 1
    for (int d = 0; d < 8; d++) {
        // 1 + exp(-o) -> slot 3 (sigmoid(o) folded into the 1/n recip below), tanh z = 2 sigmoid(2z) - 1 -> slot 2
        sfpi::dst_reg[3 * ts] = 1.0f + _slstm_exp_(-sfpi::vFloat(sfpi::dst_reg[3 * ts]));
        sfpi::vFloat z = sfpi::dst_reg[2 * ts];
        sfpi::dst_reg[2 * ts] = 2.0f * _slstm_sigmoid_(z + z) - 1.0f;
        if constexpr (first) {
            sfpi::vFloat tz = sfpi::dst_reg[2 * ts];
            sfpi::vFloat i0 = sfpi::dst_reg[0];
            sfpi::dst_reg[4 * ts] = i0;
            sfpi::dst_reg[5 * ts] = tz;
            sfpi::dst_reg[6 * ts] = 1.0f;
            sfpi::dst_reg[3 * ts] = tz * _slstm_recip_(sfpi::dst_reg[3 * ts]);
        } else {
            sfpi::vFloat lfm = _slstm_logsig_(sfpi::dst_reg[ts]) + sfpi::dst_reg[4 * ts];
            sfpi::vFloat i = sfpi::dst_reg[0];
            sfpi::vFloat mp = i;
            v_if(lfm > i) { mp = lfm; }
            v_endif;
            sfpi::dst_reg[4 * ts] = mp;
            // One of ig, fg is exp(0) = 1: e = exp(-|i - lfm|) is the other.
            sfpi::vFloat e = _slstm_exp_(-sfpi::setsgn(i - lfm, 0));
            sfpi::dst_reg[0] = 1.0f;  // ig
            sfpi::dst_reg[ts] = 1.0f;  // fg
            v_if(lfm > i) { sfpi::dst_reg[0] = e; }
            v_else { sfpi::dst_reg[ts] = e; }
            v_endif;
            sfpi::vFloat ig = sfpi::dst_reg[0];
            sfpi::vFloat fg = sfpi::dst_reg[ts];
            sfpi::vFloat c = fg * sfpi::dst_reg[5 * ts] + ig * sfpi::dst_reg[2 * ts];
            sfpi::dst_reg[5 * ts] = c;
            sfpi::vFloat n = fg * sfpi::dst_reg[6 * ts] + ig;
            v_if(n < 1.0f) { n = 1.0f; }
            v_endif;
            sfpi::dst_reg[6 * ts] = n;
            sfpi::dst_reg[3 * ts] = c * _slstm_recip_(sfpi::dst_reg[3 * ts] * n);
        }
        sfpi::dst_reg++;
    }
}

inline void slstm_gates_init_fn() { math::reset_counters(p_setrwc::SET_ABD_F); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_math_slstm_gates_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(ckernel::sfpu::slstm_gates_init_fn);
}

template <bool first>
inline void llk_math_slstm_gates() {
    _llk_math_eltwise_unary_sfpu_params_(ckernel::sfpu::calculate_slstm_gates<first>, 0, VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_MATH
