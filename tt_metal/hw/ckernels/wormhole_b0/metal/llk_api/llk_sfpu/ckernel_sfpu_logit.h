// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
// For log1p_init (programs the shared vConstFloatPrgm0..2 used below).
#include "ckernel_sfpu_log1p.h"

namespace ckernel {
namespace sfpu {

// logit(x) = log(x / (1 - x)) = log(2x) - log(2 - 2x), as a single fused SFPU op.
//
// Replaces the previous hybrid HLK (~16 chained tile ops: copy, rsub, div,
// log, mul, sub, log1p x2, abs, ltz, where, ...) with one pass.
//
// We evaluate it as
//     logit(x) = log_body(2x) - log1p(1 - 2x)
// using log(2x) == log_body(2x) and log(2 - 2x) == log1p((2 - 2x) - 1) ==
// log1p(1 - 2x).
//
// Why 2x / (1 - 2x) rather than x / (1 - x):
//
//   Near x = 0.5 (where logit'(0.5) = 4 is minimal, so a constant ~1e-7
//   absolute error -- what the old divider form produced -- costs thousands of
//   ULPs) the result is tiny and any large intermediate would lose it to
//   cancellation. With the 2x form both intermediates are *small*:
//     - log_body(2x):  2x is near 1, so the range reduction has exponent
//       k = 0 and returns poly(2x - 1) directly -- a value ~ (2x - 1), no
//       -log2 term to swamp it.
//     - log1p(1 - 2x): 1 - 2x is near 0 and exact (Sterbenz for x in [0.25,1]),
//       so log1p returns poly(1 - 2x) ~ (1 - 2x) directly.
//   Subtracting two small values has no catastrophic cancellation, giving
//     log1p(2x-1) - log1p(1-2x) = 4(x - 0.5) + O((x-0.5)^3), good to a few ULPs.
//   (Using x / (1 - x) instead would make each log ~ log(0.5) = -0.693, and
//   adding that -log2 before subtracting reintroduces the ~1e-7 error.)
//
//   log_body(2x) -- not log1p(2x - 1) -- supplies log(2x) so tiny x stays
//   accurate: log1p(2x - 1) collapses once 2x - 1 rounds to -1, whereas
//   log_body does a true exponent reduction of 2x.
//
// Both helpers are open-coded here (not pulled from the library log / log1p
// ops): inlining two different heavyweight library functions in one SFPU pass
// overflows the register file (sfpi LRA "maximum reload insns" ICE). The two
// cheap Norbert-Juffa range reductions both feed a single shared minimax
// polynomial _logit_log1p_poly (one function, two calls), which fits.
//
// vConstFloatPrgm0..2 are programmed by logit_init (== log1p_init).
//
// Endpoint / out-of-domain behavior matches torch.special.logit(eps=None):
//   x = 0   -> log(0) - log(2)     = -inf
//   x = 1   -> log(2) - log(0)     = +inf
//   x < 0   -> log(2x) of neg      =  NaN
//   x > 1   -> log1p(1 - 2x), arg(2-2x) < 0 -> NaN

// Shared minimax for log1p(m), m in [-0.25, 0.5). Identical coefficients to the
// fp32 paths of calculate_log_body and calculate_log1p_fp32.
sfpi_inline sfpi::vFloat _logit_log1p_poly(sfpi::vFloat m) {
    sfpi::vFloat s = m * m;
    sfpi::vFloat r = -0x1.92cp-5f;
    r = r * m + 0x1.b84p-4f;
    r = r * m + -0x1.0c4p-3f;
    r = r * m + 0x1.274p-3f;
    r = r * m + -0x1.55p-3f;
    r = r * m + 0x1.998p-3f;
    r = r * m + sfpi::vConstFloatPrgm1;
    r = r * m + sfpi::vConstFloatPrgm2;
    r = r * m + -0.5f;
    r = r * s + m;  // log1p(m) = m + m^2 * P(m)
    return r;
}

// FAST_APPROX is accepted for signature parity with the log1p SFPU op (so the
// shared invocation macro applies) but is unused.
template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_logit() {
#pragma GCC unroll 2
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];

        // ---- numerator: log(2x) via the log() reduction (calculate_log_body) ----
        sfpi::vFloat lognum = std::numeric_limits<float>::quiet_NaN();
        {
            sfpi::vFloat a = x + x;  // 2x, exact
            // No subnormal normalisation: subnormal inputs are outside logit's
            // useful domain and are filtered by the caller; skipping it (and the
            // log_body addexp/exexp 0 -> -inf special case) keeps this fused
            // two-log pass within the SFPU reload budget.
            sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(a) - sfpi::reinterpret<sfpi::vInt>(sfpi::vFloat(0.75f));
            e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));
            sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e) - sfpi::vConst1;
            v_if(a >= 0.0f) {
                sfpi::vFloat e_float = sfpi::int32_to_float(sfpi::abs(e), sfpi::RoundMode::NearestEven);
                e_float = sfpi::copysgn(e_float, sfpi::reinterpret<sfpi::vFloat>(e));
                lognum = e_float * sfpi::vConstFloatPrgm0 + _logit_log1p_poly(m);
            }
            v_endif;
        }

        // ---- denominator: log(2 - 2x) via the log1p(1 - 2x) reduction ----
        sfpi::vFloat logden = std::numeric_limits<float>::quiet_NaN();
        {
            sfpi::vFloat a = sfpi::vConst1 - (x + x);  // 1 - 2x, exact near 0.5
            sfpi::vFloat u = a + sfpi::vConst1;        // 2 - 2x
            v_if(u >= 0.0f) {
                sfpi::vInt e = sfpi::reinterpret<sfpi::vInt>(u) - sfpi::reinterpret<sfpi::vInt>(sfpi::vFloat(0.75f));
                e = sfpi::reinterpret<sfpi::vInt>(sfpi::setman(sfpi::reinterpret<sfpi::vFloat>(e), 0));
                sfpi::vFloat m = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(a) - e);
                sfpi::vFloat neg_four = -4.0f;
                sfpi::vFloat s = sfpi::reinterpret<sfpi::vFloat>(sfpi::reinterpret<sfpi::vInt>(neg_four) - e);
                sfpi::vFloat neg_quarter = -0.25f;
                sfpi::vFloat neg1 = sfpi::vConstNeg1;
                // t = 2^-k - 1
                sfpi::vFloat t =
                    __builtin_rvtt_sfpmad(neg_quarter.get(), s.get(), neg1.get(), sfpi::SFPMAD_MOD1_OFFSET_NONE);
                m = m + t;
                sfpi::vFloat e_float = sfpi::int32_to_float(sfpi::abs(e), sfpi::RoundMode::NearestEven);
                e_float = sfpi::copysgn(e_float, sfpi::reinterpret<sfpi::vFloat>(e));
                logden = e_float * sfpi::vConstFloatPrgm0 + _logit_log1p_poly(m);
                // No u == +inf / NaN guard needed: for valid logit inputs
                // x in (0, 1), u = 2 - 2x lies in (0, 2) and is finite. The
                // u == 0 case (x == 1) reduces to -inf through the same bit
                // path, so logit(1) = log(2) - (-inf) = +inf falls out directly.
            }
            v_endif;
        }

        sfpi::vFloat result = lognum - logden;

        // log_body(2x) omits the 0 -> -inf special case (register budget), so it
        // saturates to a large negative value at 2x == 0; patch logit(0) = -inf
        // here. x < 0 is out of domain and correctly stays NaN via lognum.
        v_if(x == 0.0f) { result = -std::numeric_limits<float>::infinity(); }
        v_endif;

        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::NearestEven);
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, bool FAST_APPROX, bool is_fp32_dest_acc_en>
inline void logit_init() {
    // log(2x) and log1p(1 - 2x) reductions read the same three programmable
    // float constants; log1p_init programs them. _logit_log1p_poly always uses
    // the fp32 minimax (the bf16 dest path computes in fp32 and only rounds the
    // final result), so program the fp32 constant set regardless of dest mode.
    log1p_init<APPROXIMATION_MODE, FAST_APPROX, /*is_fp32_dest_acc_en=*/true>();
}

}  // namespace sfpu
}  // namespace ckernel
