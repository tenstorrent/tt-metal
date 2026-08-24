// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel::sfpu
{

// LICENSED gelu-255 semantic body (owner ratification 2026-08-24 item 2:
// "gelu at the hand kernel's 255-ulp contract" — equal-or-better error than
// the hand kernel on the row's golden domain, never worse).
//
// THE MEASURED HAND CONTRACT (laneGI accuracy oracle, exact port of
// calculate_gelu_piecewise over all finite bf16 vs the fp64 golden):
//   max_abs 0.00776871 (x ~ 2.766, bf16 store quantization)
//   max_pure_bf16_ulp 253.194 (x ~ -7.03, the torch-saturation flush) —
//   the SAME 253.19366 the fitter board's ttnn_pure records: the "255-ulp
//   contract" is real on this row's own domain (untruncated Gaussian), not
//   a [-10,10] artifact.  253 bf16 ulp ~ 99% relative slack domain-wide.
//
// This licensed arm spends that slack on POLYNOMIAL DEPTH, keeping the hand
// kernel's own region structure (the +2.54%-parity fresh body's shape):
//   x <= -5.54259443 : 0                     (the hand flush, unchanged)
//   exp region       : x * 2^(x2*-0.72134752044 + [127]) with a DEG-1
//                      fractional refine (m = 1 + 0.92355*t, setexp-safe
//                      m in [1, 1.924)) and a LINEAR H-correction — drops
//                      the deg-2 frac, deg-3 corr, and the 2-op deep-tail
//                      grid snap; composed max rel err 6.70%, max abs
//                      1.86e-4 (vs 99% / 0.0078 budgets)
//   core             : deg-11 odd Phi fit (was deg-13), max abs 8.50e-5
//                      (a deg-9 fit measured 0.00819 > the 0.00777 abs bar
//                      at the core's top and was REJECTED by the oracle)
//   x >= 2.78125     : x                     (identity, unchanged)
// Fits: laneGI-evidence-20260824/accuracy-oracle/fit_gelu255*.out; proof of
// composite dominance (max-abs AND max-pure-ulp <= hand on all finite
// bf16): gelu255_verify.c in the same directory.
template <int ITERATIONS>
__attribute__((noinline)) void calculate_gelu_255_licensed_cpp()
{
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; ++d)
    {
        const sfpi::vFloat x  = sfpi::dst_reg[0];
        const sfpi::vFloat x2 = x * x;
        sfpi::vFloat r        = 0.0f;
        v_if (x > -5.54259443f)
        {
            const sfpi::vFloat xlog2 = x2 * -0.72134752044f + 127.0f;
            // Moroz fixed-point encoding (the hand kernel's own
            // _float_to_int32_for_exp_21f_ dataflow, stated typed).
            const sfpi::vInt zi     = sfpi::shft(sfpi::exman(xlog2, sfpi::MantissaMode::ImplicitOne), sfpi::exexp(xlog2), sfpi::ShiftMode::Logical);
            const sfpi::vFloat z    = sfpi::as<sfpi::vFloat>(zi);
            const sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            // deg-1 refine of 2^t over the integer mantissa domain
            // (0.92355 / 2^23 = 1.1009574e-07); m in [1, 1.924) keeps
            // setexp mantissa-reuse semantics exact.
            const sfpi::vFloat m    = frac * 1.1009574e-07f + 1.0f;
            const sfpi::vFloat e2   = sfpi::setexp(m, sfpi::exexp(z, sfpi::ExponentMode::Biased));
            const sfpi::vFloat corr = x * 0.018694703f + 0.17122011f;
            r                       = x * (e2 * corr);
            v_and(x >= -3.125f);
            sfpi::vFloat odd = ((((x2 * -1.1136863e-06f + 4.6201898e-05f) * x2 + -0.00083983334f) * x2 + 0.00900611f) * x2 + -0.06519476f) * x2 + 0.398352f;
            r                = x * (x * odd + 0.5f);
            v_and(x >= 2.78125f);
            r = x;
        }
        v_endif;
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(r, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
