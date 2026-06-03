// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel_sfpu_exp.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// Computes 2^x directly in base-2.
//
// Decompose x = n + f, where n = round(x) and f = x - n ∈ [-0.5, 0.5].
// Then 2^x = 2^n * 2^f, where:
//   - 2^n is realised by adding n to the IEEE-754 exponent of 2^f via setexp.
//   - 2^f is approximated by a degree-6 minimax polynomial on [-0.5, 0.5].
//
// This bypasses the redundant `* ln(2)` / `* 1/ln(2)` round-trip that the
// previous implementation paid for by routing through `exp(x * ln(2))`.
//
// Caller MUST clamp x to the safe range (~[-127, 128)) before calling this
// helper; the public `_calculate_exp2_` wrapper handles the clamping and
// special-case dispatch.
sfpi_inline sfpi::vFloat _sfpu_exp2_core_unsafe_(sfpi::vFloat x) {
    // Step 1: split x into integer part n (as int32) and fractional part f.
    // We use the same round-to-nearest-even trick as _sfpu_round_to_nearest_int32_
    // (see ckernel_sfpu_exp.h) to extract both n_float and n_int in one shot.
    sfpi::vInt n_int;
    sfpi::vFloat n_float = _sfpu_round_to_nearest_int32_(x, n_int);
    sfpi::vFloat f = x - n_float;
    /*
    sfpi::vInt i = sfpi::float_to_int16(x);
    sfpi::vFloat j = sfpi::int32_to_float(i);
    sfpi::vFloat f = x - j;
    i = sfpi::abs(i);
    i = sfpi::reinterpret<sfpi::vInt>(sfpi::copysgn(sfpi::reinterpret<sfpi::vFloat>(i), x));
    */

    // Step 2: 2^f via degree-6 relative minimax polynomial on [-0.5, 0.5].
    // Degree-5 Taylor reaches ~39 FP32 ULP near the interval endpoints.
    //   2^f ≈ c0 + c1·f + c2·f² + c3·f³ + c4·f⁴ + c5·f⁵ + c6·f⁶
    sfpi::vFloat p = PolynomialEvaluator::eval(
        f,
        sfpi::vConst1,    // c0
        0x1.62e430p-1f,   // c1
        0x1.ebfbdap-3f,   // c2
        0x1.c6aed4p-5f,   // c3
        0x1.3b2dbcp-7f,   // c4
        0x1.5f456ap-10f,  // c5
        0x1.41d334p-13f   // c6
    );
    // default inf
    // v_if () {
    //} v_endif;

    // Step 3: scale by 2^n via direct exponent injection: setexp(p, exexp(p) + n).
    sfpi::vInt p_exp = sfpi::exexp(p, sfpi::ExponentMode::NoDebias);
    // float_to_uint8(int32_to_float(p_exp) + j)
    sfpi::vInt new_exp = p_exp + n_int;
    return sfpi::setexp(p, new_exp);
}

// FP32 accurate path for 2^x with branch-free saturation and explicit NaN override.
//
// Thresholds are applied directly in the base-2 domain (x = log2 of result):
//   - x ≥ 128 would overflow biased exponent (≥ 255) → clamp lands on the +inf encoding.
//   - x ≤ -127 underflows into the denormal range → setexp produces a subnormal that
//     flushes to zero, matching IEEE FTZ semantics used elsewhere in the SFPU stack.
//
// vec_min_max handles ±inf naturally (clamped to ±boundary) but propagates NaN
// unreliably under -ffast-math, so we re-detect NaN on the *original* x using the
// integer-exponent pattern from `_calculate_isnan_` (exexp default mode returns the
// unbiased exponent, where 128 = biased 255 = inf-or-NaN).
sfpi_inline sfpi::vFloat _sfpu_exp2_fp32_accurate_(sfpi::vFloat x) {
    constexpr float OVERFLOW_THRESHOLD = 128.0f;
    constexpr float UNDERFLOW_THRESHOLD = -127.0f;

    // Clamp x to [-127, 128]; out-of-range inputs land on ±boundary and pick up
    // the exact +inf / +0 encodings via setexp saturation in the core.
    // vec_min_max takes non-const lvalue references, so the thresholds must be
    // named locals rather than temporaries.
    sfpi::vFloat x_clamped = x;
    sfpi::vFloat underflow_lim = UNDERFLOW_THRESHOLD;
    sfpi::vFloat overflow_lim = OVERFLOW_THRESHOLD;
    sfpi::vec_min_max(underflow_lim, x_clamped);
    sfpi::vec_min_max(x_clamped, overflow_lim);

    sfpi::vFloat result = _sfpu_exp2_core_unsafe_(x_clamped);

    // NaN override: default sfpi::exexp returns the *unbiased* exponent, so both
    // inf and NaN have exp == 128, distinguished by mantissa (matches the canonical
    // pattern in `_calculate_isnan_`). Only true NaN needs an override — ±inf inputs
    // already produce the correct ±inf / 0 result via the clamp.
    sfpi::vInt nan_exp = sfpi::exexp(x);
    sfpi::vInt nan_man = sfpi::exman(x);
    v_if(nan_exp == 128 && nan_man != 0) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return result;
}

// BF16 path: branch-free saturation using the mantissa-as-fractional-part trick
// from the production `_sfpu_exp_21f_bf16_` kernel — but specialised for exp2 by
// skipping the `* (1/ln2)` multiply (we are already in base-2). vec_min_max clamps
// xlog2 to [0, 255], so the natural saturation of setexp + the final bf16 round
// give the correct overflow → +inf and underflow → 0 boundary encodings for free.
//
// NaN: ttnn.bfloat16 host-side packing already collapses NaN → +inf before the
// tensor ever reaches the SFPU (see the "NaN is packed as inf for ttnn.bfloat16"
// xfails on fmod / remainder / where / rdiv), so a device-side NaN guard here
// would be dead code.
sfpi_inline sfpi::vFloat _sfpu_exp2_bf16_(sfpi::vFloat x) {
    // Map x → xlog2 such that 2^x has biased exponent floor(xlog2) and the
    // fractional mantissa supplies the (xlog2 - floor) refinement.
    sfpi::vFloat xlog2 = x + 127.f;

    // Clamp to [0, 255]. Boundary inputs land on the +inf / +0 encodings after
    // setexp + bf16 round.
    sfpi::vFloat threshold_low = 0.f;
    sfpi::vFloat threshold_high = sfpi::vFloat(255.f);
    sfpi::vec_min_max(threshold_low, xlog2);
    sfpi::vec_min_max(xlog2, threshold_high);

    // Decompose xlog2 in [0, 255] into:
    //   exponential_part = floor(xlog2)             (integer in [0, 255])
    //   fractional_part  = (xlog2 - floor) * 2^23   (integer in [0, 2^23))
    sfpi::vInt z = _float_to_int32_for_exp_21f_(xlog2);

    sfpi::vInt exponential_part = sfpi::exexp(sfpi::reinterpret<sfpi::vFloat>(z), sfpi::ExponentMode::NoDebias);
    sfpi::vInt fractional_part = sfpi::exman(sfpi::reinterpret<sfpi::vFloat>(z));

    sfpi::vFloat frac = sfpi::int32_to_float(fractional_part, sfpi::RoundMode::NearestEven);

    // Refine 2^x_f on x_f to [0, 2^23). Same minimax coefficients as the
    // production exp_21f kernel (≤ 3 fp32 ULP, well under 1 bf16 ULP).
    frac = PolynomialEvaluator::eval(frac, 1.0017248f, 7.839635491371155e-08f, 4.791750143340323e-15f);

    // Recombine: 2^x = (1.frac_mantissa) * 2^(exponential_part - 127).
    sfpi::vFloat y = sfpi::setexp(frac, exponential_part);

    // SFPSTORE truncates fp32→bf16; round explicitly so the bf16 result matches
    // a faithful nearest-even rounding of the fp32 mathematical value, and so
    // that the saturation tricks above (overflow → +inf, underflow → 0) land
    // on the correct bf16 encoding.
    return sfpi::convert<sfpi::vFloat16b>(y, sfpi::RoundMode::NearestEven);
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en = false, int ITERATIONS = 8>
inline void calculate_exp2() {
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        if constexpr (is_fp32_dest_acc_en) {
            sfpi::dst_reg[0] = _sfpu_exp2_fp32_accurate_(v);
        } else {
            sfpi::dst_reg[0] = _sfpu_exp2_bf16_(v);
        }

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void exp2_init() {
    // The optimised exp2 implementation works directly in base-2 and does not
    // require any program-constant register. Kept as a no-op to preserve the
    // public init API (callers in `exp2_init<APPROX>::call` and the LLK test
    // harness still invoke this).
}

}  // namespace ckernel::sfpu
