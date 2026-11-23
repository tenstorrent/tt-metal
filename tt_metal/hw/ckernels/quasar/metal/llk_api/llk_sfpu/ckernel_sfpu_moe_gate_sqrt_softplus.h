// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// ======================================================================
// MoE gate (DeepSeek-V4) router activation fused with the expert-bias add, fp32 DEST only:
//   score = sqrt(softplus(x))        (beta = 1, threshold = 20, as torch.nn.functional.softplus)
//   key   = score + bias
// Reads x from tile in0 and the bias from tile in1, writes the score over in0 and the key over in1.
//
// softplus(x) = max(x, 0) + f(|x|) with f(a) = ln(1 + exp(-a)) = e * g(e), e = exp(-a) in (0, 1],
// g(e) = log1p(e) / e in [ln 2, 1]:
//   e = 2^-w, w = a * log2(e): w = k + r with k = round(w) (16-bit sign-magnitude, saturating) and
//       r in [-1/2, 1/2]; 2^-r by a degree-4 polynomial (rel. error 2.9e-6); the exponent of the
//       result is lowered by k, and underflow (k past the fp32 exponent range, incl. +-inf/huge |x|) is 0.
//   g(e) by a degree-6 polynomial with g(0) = 1 pinned (rel. error 2.8e-6), so f keeps its relative
//       accuracy for every a, including the far tail where f is the whole softplus of a negative x.
// Max relative error of softplus ~5e-6 over all inputs (the bf16 polynomial the stock softplus runs
// without INP_FLOAT32 has 1.6e-3 relative error for x >= 0 and drops f entirely for |x| > 5).
// Above the threshold (x > 20) f(x) < 2.1e-9 is below half an ulp of x, so max(x, 0) + f == x exactly
// and no threshold branch is needed. The sqrt is Blackhole's stock 23-bit SFPU sqrt (_moe_gate_sqrt_ below; Quasar's
// own sqrt is the approximate SFPNONLINEAR SQRT_MODE).
//
// Polynomial coefficients: c0 = 1 (constant register), the high-order ones are fp16-exact (one SFPLOADI).
// ======================================================================
constexpr float MOE_GATE_LOG2E = 1.4426950408889634f;

constexpr float MOE_GATE_EXP2N_C1 = -0.693123996257782f;
constexpr float MOE_GATE_EXP2N_C2 = 0.24024111032485962f;
constexpr float MOE_GATE_EXP2N_C3 = -0.055908203125f;   // fp16-exact
constexpr float MOE_GATE_EXP2N_C4 = 0.00958251953125f;  // fp16-exact

constexpr float MOE_GATE_LOG1P_C1 = -0.4999009072780609f;
constexpr float MOE_GATE_LOG1P_C2 = 0.33141666650772095f;
constexpr float MOE_GATE_LOG1P_C3 = -0.2362060546875f;  // fp16-exact
constexpr float MOE_GATE_LOG1P_C4 = 0.151123046875f;    // fp16-exact
constexpr float MOE_GATE_LOG1P_C5 = -0.06787109375f;    // fp16-exact
constexpr float MOE_GATE_LOG1P_C6 = 0.01458740234375f;  // fp16-exact

sfpi_inline sfpi::vFloat _moe_gate_softplus_(const sfpi::vFloat x) {
    sfpi::vFloat w = sfpi::setsgn(x, 0) * MOE_GATE_LOG2E;

    sfpi::vSMag16 sm = sfpi::convert<sfpi::vSMag16>(w, sfpi::RoundMode::Nearest);
    sfpi::vFloat r = w - sfpi::convert<sfpi::vFloat>(sm, sfpi::RoundMode::Nearest);

    // 2^-r, r in [-1/2, 1/2]
    sfpi::vFloat p = r * MOE_GATE_EXP2N_C4 + MOE_GATE_EXP2N_C3;
    p = p * r + MOE_GATE_EXP2N_C2;
    p = p * r + MOE_GATE_EXP2N_C1;
    p = p * r + 1.0f;

    // e = 2^-k * p; w >= 0, so the sign-magnitude k is its two's complement value.
    sfpi::vInt k = sfpi::as<sfpi::vInt>(sm);
    sfpi::vInt new_exp = sfpi::exexp(p, sfpi::ExponentMode::Biased) - k;
    sfpi::vFloat e = 0.0f;
    v_if(new_exp > 0) { e = sfpi::setexp(p, new_exp); }
    v_endif;

    // f = e * g(e)
    sfpi::vFloat g = e * MOE_GATE_LOG1P_C6 + MOE_GATE_LOG1P_C5;
    g = g * e + MOE_GATE_LOG1P_C4;
    g = g * e + MOE_GATE_LOG1P_C3;
    g = g * e + MOE_GATE_LOG1P_C2;
    g = g * e + MOE_GATE_LOG1P_C1;
    g = g * e + 1.0f;

    return g * e + sfpi::max(x, 0.0f);
}

// Blackhole's _calculate_sqrt_body_<false, false, false> (Algorithm SQRT_23-bits, Kokosinski et al., Numerical
// Algorithms 2024, https://doi.org/10.1007/s11075-024-01932-7), verbatim: the non-approximate, non-reciprocal path
// with its +-0 / inf / negative edge handling. Reads the seed and refinement constants that
// moe_gate_sqrt_softplus_init programs into vConstIntPrgm0 / vConstFloatPrgm1 / vConstFloatPrgm2.
sfpi_inline sfpi::vFloat _moe_gate_sqrt_(const sfpi::vFloat x) {
    sfpi::vInt i = sfpi::as<sfpi::vInt>(sfpi::as<sfpi::vUInt>(x) >> 1);
    sfpi::vFloat y = sfpi::as<sfpi::vFloat>(sfpi::vConstIntPrgm0 - i);

    sfpi::vFloat xy = x * y;
    sfpi::vFloat negative_y = -y;
    sfpi::vFloat c = negative_y * xy;
    sfpi::vFloat infinity = sfpi::sFloat16b(std::numeric_limits<float>::infinity());
    sfpi::vInt infinity_bits = sfpi::as<sfpi::vInt>(infinity);
    y = y * (sfpi::vConstFloatPrgm1 + c * (sfpi::vConstFloatPrgm2 + c));
    xy = x * y;
    negative_y = -y;
    sfpi::vFloat one_minus_xyy = 1.0f + (negative_y * xy);

    sfpi::vFloat half_xy = 0.5f * xy;
    // If x == inf, skip to avoid y = inf - inf = nan; y is already inf.
    v_if(sfpi::as<sfpi::vInt>(x) < infinity_bits) { y = one_minus_xyy * half_xy + xy; }
    v_endif;

    // sqrt(+/-0) = +/-0: return x, because the refinement cannot produce a signed zero. The shift drops the sign so
    // -0.0 and +0.0 test alike.
    v_if((sfpi::as<sfpi::vInt>(x) << 1) == 0) { y = x; }
    v_elseif(x < 0.0f) { y = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;

    return y;
}

// Programs the SQRT_23-bits constants _moe_gate_sqrt_ reads (Blackhole's sqrt_init<false>).
inline void moe_gate_sqrt_softplus_init() {
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
    sfpi::vConstIntPrgm0 = 0x5f1110a0;
    sfpi::vConstFloatPrgm1 = 2.2825186f;
    sfpi::vConstFloatPrgm2 = 2.2533049f;
}

template <int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_moe_gate_sqrt_softplus_add_bias(
    const std::uint32_t dst_index_x, const std::uint32_t dst_index_bias, const std::uint32_t /*dst_index_out*/) {
    // No bf16 round-to-nearest step: with a 16-bit DEST the score and key would be stored truncated.
    static_assert(is_fp32_dest_acc_en, "moe_gate_sqrt_softplus_add_bias requires fp32 DEST (fp32_dest_acc_en)");
    constexpr std::uint32_t dst_tile_size_sfpi = 32;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[dst_index_x * dst_tile_size_sfpi];
        sfpi::vFloat score = _moe_gate_sqrt_(_moe_gate_softplus_(x));
        sfpi::dst_reg[dst_index_x * dst_tile_size_sfpi] = score;
        sfpi::vFloat bias = sfpi::dst_reg[dst_index_bias * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_index_bias * dst_tile_size_sfpi] = score + bias;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
