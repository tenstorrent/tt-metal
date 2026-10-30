// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"

// Lean bf16-dest forms of the four fused binary GLU activations used by the routed-expert FFN
// (deepseek_prefill unified_routed_expert_ffn fused_swiglu.cpp). They compute the same functions as
// calculate_silu_glu, calculate_situ_glu, calculate_swiglu (moe_gpt swiglu_sfpu.h) and
// calculate_clamped_silu_glu, with:
//
//   * exp from the same exp_21f algorithm as the baseline sigmoid (Moroz et al., 21-bit), its three
//     refinement coefficients held in vConstFloatPrgm0..2 (glu_fast_init) instead of being reloaded per row;
//   * one Newton step on approx_recip, as the baseline bf16 sigmoid, without the NaN branch: every
//     denominator here is finite and >= 1 by construction (the exp inputs are clamped);
//   * SiTU-GLU rewritten so both tanh halves and the sigmoid share ONE reciprocal and use no polynomial:
//       u = exp(-g/2), v = exp(-2*up/25)
//       4*tanh(g/4) * sigmoid(g) = 4 * (1-u) / ((1+u)(1+u^2))
//       25*tanh(up/25)          = 25 * (1-v) / (1+v)
//       result                  = 100 * (1-u)(1-v) / ((1+u)(1+u^2)(1+v))
//     which is more accurate than the degree-6 tanh polynomial it replaces.
//
// bf16 dest (fp32_dest_acc_en = false) only; the fp32-dest callers keep the original routines.
// Contract: glu_fast_init() programs vConstFloatPrgm0..2 and nothing may reprogram them before the calls.

namespace ckernel::sfpu {

constexpr float GLU_FAST_ONE_LN2 = 1.4426950216293334961f;

// exp(x) given xlog2 = x / ln2 + 127, already clamped into [0, 250]: the exp_21f body of
// _sfpu_exp_21f_bf16_ after its clamp, with c0..c2 read from the programmable constants.
sfpi_inline sfpi::vFloat _glu_fast_exp_(sfpi::vFloat xlog2) {
    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);
    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, sfpi::vConstFloatPrgm0, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm2);
    return sfpi::setexp(frac, exponential_part);
}

// 1/d for finite d >= 1: approx_recip plus one Newton step (t is negated, as in sfpu_reciprocal_iter).
sfpi_inline sfpi::vFloat _glu_fast_recip_(sfpi::vFloat d) {
    sfpi::vFloat y = sfpi::approx_recip(d);
    sfpi::vFloat t = d * y - 2.0f;
    return y * -t - 0.0f;
}

template <int ITERATIONS = 8>
inline void calculate_situ_glu_fast(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    // exp(-g/2) and exp(-2*up/25) in the exp_21f domain: x * k + 127.
    const sfpi::vFloat k_gate = -0.5f * GLU_FAST_ONE_LN2;
    const sfpi::vFloat k_up = -0.08f * GLU_FAST_ONE_LN2;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        // Clamp to [0, 150]: u, v <= 2^23 keeps the denominator below 2^92. The cut-offs are
        // g < -32 (|result| < 4e-14 * |up_half|) and up < -200 (tanh(up/25) = -1 in fp32).
        sfpi::vFloat xg = sfpi::clamp(g * k_gate + 127.0f, 0.0f, 150.0f);
        sfpi::vFloat xu = sfpi::clamp(up * k_up + 127.0f, 0.0f, 150.0f);
        sfpi::vFloat u = _glu_fast_exp_(xg);
        sfpi::vFloat v = _glu_fast_exp_(xu);

        sfpi::vFloat den = (u + 1.0f) * (u * u + 1.0f);
        den = den * v + den;
        sfpi::vFloat num = (u * -100.0f + 100.0f) * (1.0f - v);

        sfpi::vFloat result = num * _glu_fast_recip_(den);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// gate_clamped * sigmoid(alpha * gate_clamped) * (up_clamped + up_offset), gate clamped above at `limit`,
// up to [-limit, limit]. SwiGLU-OAI: alpha 1.702, limit 7, up_offset 1. Clamped SiLU-GLU: 1, 10, 0.
template <int ITERATIONS, int ALPHA_MILLI, int LIMIT, int UP_OFFSET>
inline void calculate_clamped_sigmoid_glu_fast(
    const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    constexpr float alpha = ALPHA_MILLI / 1000.0f;
    constexpr float limit = static_cast<float>(LIMIT);
    const sfpi::vFloat k = -alpha * GLU_FAST_ONE_LN2;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        g = sfpi::min(g, limit);
        up = sfpi::clamp(up, -limit, limit);
        if constexpr (UP_OFFSET != 0) {
            up = up + static_cast<float>(UP_OFFSET);
        }

        // g <= limit keeps xlog2 >= 127 - alpha * limit / ln2 > 0, so only the top needs a clamp;
        // 250 keeps exp(-alpha * g) finite (<= 2^124), so 1 + e never reaches inf.
        sfpi::vFloat xg = sfpi::min(g * k + 127.0f, 250.0f);
        sfpi::vFloat sig = _glu_fast_recip_(_glu_fast_exp_(xg) + 1.0f);

        sfpi::vFloat result = (g * sig) * up;
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// silu(gate) * up = gate * sigmoid(gate) * up (plain SwiGLU: DeepSeek-V3 / Kimi-K2.7 / GLM / Mistral).
// Same exp_21f + one Newton reciprocal as calculate_silu_glu's bf16 arm, without the reciprocal's NaN branch
// (1 + exp(-g) is finite and >= 1: the exp input is clamped to [0, 250]).
template <int ITERATIONS = 8>
inline void calculate_silu_glu_fast(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;
    const sfpi::vFloat k = -GLU_FAST_ONE_LN2;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat g = sfpi::dst_reg[gate_tile_idx * dst_tile_size];

        // g > 88 -> exp(-g) underflows to 2^-127 (sigmoid 1); g < -117 -> 2^123 (silu ~ 0).
        sfpi::vFloat xg = sfpi::clamp(g * k + 127.0f, 0.0f, 250.0f);
        sfpi::vFloat sig = _glu_fast_recip_(_glu_fast_exp_(xg) + 1.0f);

        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat result = (g * sig) * up;
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <int ITERATIONS = 8>
inline void calculate_swiglu_oai_fast(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    calculate_clamped_sigmoid_glu_fast<ITERATIONS, 1702, 7, 1>(gate_tile_idx, up_tile_idx, out_tile_idx);
}

template <int ITERATIONS = 8>
inline void calculate_clamped_silu_glu_fast(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    calculate_clamped_sigmoid_glu_fast<ITERATIONS, 1000, 10, 0>(gate_tile_idx, up_tile_idx, out_tile_idx);
}

inline void glu_fast_init() {
    sfpi::vConstFloatPrgm0 = EXP_21F_BF16_C0;
    sfpi::vConstFloatPrgm1 = EXP_21F_BF16_C1;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}

}  // namespace ckernel::sfpu
