// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_silu.h"
#include "ckernel_sfpu_clamped_silu_glu.h"
#include "ckernel_sfpu_situ_glu.h"
#include "ckernel_sfpu_softcap.h"
#include "ckernel_sfpu_tanh.h"

// Activation routines of unified_routed_expert_moe (Silu, ClampedSiluGlu, SwiGluOai, SituGlu) for the
// bf16-dest path, with the loop-invariant constants loaded ONCE per call instead of once per row
// (SituGlu also derives its sigmoid from its own tanh; see that section).
//
// The stock routines spell their fp32 constants as literals inside the row loop, and sfpi
// materialises each one with an SFPLOADI pair on every row it is used: 9 of the 33 instructions of a
// silu row, ~25 of the ~72 of a SiTU-GLU row. Here the same constants sit in LRegs declared before
// the loop, or in vConstFloatPrgm1/2 programmed by the matching init. Every arithmetic operation,
// its operand order and every constant value are those of the stock routine, so the results are
// bit-identical (only SwiGluOai's v_if clamps are spelled min/clamp, equal for every non-NaN input),
// except SituGlu's sigmoid.
//
// fp32-dest mode is not specialised: it forwards to the stock routine, and the matching init then
// runs the stock init as well.
//
// Only unified_routed_expert_ffn's compute kernel includes this header; the stock routines it
// mirrors are untouched.

namespace ckernel::sfpu {

// exp_21f constants of _sfpu_exp_21f_bf16_ (same values, see ckernel_sfpu_exp.h).
constexpr float REA_ONE_LN2 = 1.4426950216293334961f;

// vConstFloatPrgm layout the rea_* sigmoid users program in their init:
//   Prgm0 = 2.0f        (sfpu_reciprocal_iter, as sfpu_reciprocal_init sets it)
//   Prgm1 = 1 / ln2     (exp_21f argument scale)
//   Prgm2 = EXP_21F_BF16_C0
inline void _rea_sigmoid_consts_init_() {
    sfpi::vConstFloatPrgm0 = 2.0f;
    sfpi::vConstFloatPrgm1 = REA_ONE_LN2;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C0;
}

// _sfpu_exp_21f_bf16_<true>(val) with 1/ln2 and c0 read from Prgm1/Prgm2 and c1/c2 from LRegs.
sfpi_inline sfpi::vFloat _rea_exp_21f_(sfpi::vFloat val, sfpi::vFloat c1, sfpi::vFloat c2) {
    sfpi::vFloat xlog2 = (val * sfpi::vConstFloatPrgm1 + 127.f);
    xlog2 = sfpi::clamp(xlog2, 0.0f, 255.0f);

    sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));

    sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
    sfpi::vMag fractional_part = sfpi::exman(z);

    sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(fractional_part, sfpi::RoundMode::Nearest);
    frac = PolynomialEvaluator::eval(frac, sfpi::vConstFloatPrgm2, c1, c2);

    return sfpi::setexp(frac, exponential_part);
}

// _sfpu_sigmoid_<false>(x): 1 / (1 + exp_21f(-x)), one Newton step (Prgm0 = 2.0).
sfpi_inline sfpi::vFloat _rea_sigmoid_(sfpi::vFloat x, sfpi::vFloat c1, sfpi::vFloat c2) {
    sfpi::vFloat exp_neg_x = _rea_exp_21f_(-x, c1, c2);
    sfpi::vFloat denominator = 1.0f + exp_neg_x;
    return sfpu_reciprocal_iter<1>(denominator);
}

// ---------------------------------------------------------------- Silu (calculate_silu)
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8>
inline void calculate_rea_silu() {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_silu<is_fp32_dest_acc_en, ITERATIONS>();
    } else {
        sfpi::vFloat c1 = EXP_21F_BF16_C1;
        sfpi::vFloat c2 = EXP_21F_BF16_C2;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat result = x * _rea_sigmoid_(x, c1, c2);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en>
inline void rea_silu_init() {
    silu_init<APPROXIMATION_MODE>();
    if constexpr (!is_fp32_dest_acc_en) {
        _rea_sigmoid_consts_init_();
    }
}

// ---------------------------------------------------------------- ClampedSiluGlu (calculate_clamped_silu_glu)
template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = ClampedSiluGluConfigDsV4>
inline void calculate_rea_clamped_silu_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_clamped_silu_glu<is_fp32_dest_acc_en, ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    } else {
        constexpr float limit = Config::limit;
        constexpr uint dst_tile_size = 32;
        sfpi::vFloat c1 = EXP_21F_BF16_C1;
        sfpi::vFloat c2 = EXP_21F_BF16_C2;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
            sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

            gate = sfpi::min(gate, limit);
            up = sfpi::clamp(up, -limit, limit);

            sfpi::vFloat result = gate * _rea_sigmoid_(gate, c1, c2) * up;
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
            sfpi::dst_reg++;
        }
    }
}

template <bool is_fp32_dest_acc_en>
inline void rea_clamped_silu_glu_init() {
    clamped_silu_glu_init();
    if constexpr (!is_fp32_dest_acc_en) {
        _rea_sigmoid_consts_init_();
    }
}

// ---------------------------------------------------------------- SwiGluOai
// Same math as calculate_swiglu (ttnn/.../moe_gpt/device/kernels/swiglu_sfpu.h, SwiGLUConfigGPTOSS):
//   (clamp(up, -L, L) + 1) * (g * sigmoid(alpha * g)),  g = min(gate, L)
struct ReaSwiGluOaiConfig {
    static constexpr float alpha = 1.702f;
    static constexpr float clamp_limit = 7.0f;
};

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = ReaSwiGluOaiConfig>
inline void calculate_rea_swiglu_oai(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float clamp_limit = Config::clamp_limit;
    constexpr uint dst_tile_size = 32;
    sfpi::vFloat alpha = Config::alpha;
    sfpi::vFloat c1 = EXP_21F_BF16_C1;
    sfpi::vFloat c2 = EXP_21F_BF16_C2;
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        gate = sfpi::min(gate, clamp_limit);
        up = sfpi::clamp(up, -clamp_limit, clamp_limit);
        up = up + 1.0f;

        sfpi::vFloat alpha_gate = gate * alpha;
        sfpi::vFloat sig;
        if constexpr (is_fp32_dest_acc_en) {
            sig = sfpu_reciprocal_iter<2>(1.0f + _sfpu_exp_accurate_<true>(-alpha_gate));
        } else {
            sig = _rea_sigmoid_(alpha_gate, c1, c2);
        }

        sfpi::vFloat glu = gate * sig;
        sfpi::vFloat result = up * glu;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en>
inline void rea_swiglu_oai_init() {
    if constexpr (is_fp32_dest_acc_en) {
        sfpu_reciprocal_init<false>();
    } else {
        _rea_sigmoid_consts_init_();
    }
}

// ---------------------------------------------------------------- SituGlu (calculate_situ_glu)
// tanh_init owns all three vConstFloatPrgm registers here (the top tanh coefficients), so only the
// three remaining tanh coefficients move out of the loop. Each row evaluates the polynomial twice
// (tanh of gate/beta_gate and of up/beta_up), so those three LRegs save 12 SFPLOADIs per row.
//
// Unlike the other variants this one is NOT bit-identical to the stock routine: the sigmoid is derived
// from the gate softcap's own tanh instead of exp_21f + a guarded Newton reciprocal (see below).

// _sfpu_softcap_ without the final * beta: the stock non-approx tanh polynomial of x * inv_beta.
sfpi_inline sfpi::vFloat _rea_tanh_scaled_(
    sfpi::vFloat x, sfpi::vFloat inv_beta, sfpi::vFloat t1, sfpi::vFloat t2, sfpi::vFloat t3) {
    sfpi::vFloat xs = x * inv_beta;
    sfpi::vFloat val = sfpi::abs(xs);
    sfpi::vFloat result = PolynomialEvaluator::eval(
        val, 0.0f, t1, t2, t3, sfpi::vConstFloatPrgm2, sfpi::vConstFloatPrgm1, sfpi::vConstFloatPrgm0);
    result = sfpi::min(result, 1.0f);
    return sfpi::copysgn(result, xs);
}

// sigmoid(g) from t = tanh(g/4), the value the gate softcap already holds (beta_gate = 4):
//   sigmoid(g) = 1/2 + tanh(g/2)/2  and  tanh(2a) = 2 tanh(a) / (1 + tanh(a)^2)  with a = g/4
//   => sigmoid(g) = 1/2 + t / (1 + t^2)
// 1 + t^2 lies in [1, 2] (t is clamped to [-1, 1]), so the reciprocal needs no NaN/inf guard and one
// Newton step from approx_recip, written with the hardware -1 constant (y' = y - (d*y - 1)*y), since
// tanh_init holds the vConstFloatPrgm0 the stock step reads its 2.0 from. Its error is the tanh
// polynomial's times (1 - t^2)/(1 + t^2)^2 <= 1, which vanishes as |g| grows.
sfpi_inline sfpi::vFloat _rea_sigmoid_from_tanh_quarter_(sfpi::vFloat t) {
    sfpi::vFloat d = t * t + 1.0f;
    sfpi::vFloat y = sfpi::approx_recip(d);
    sfpi::vFloat e = d * y - 1.0f;
    y = y - e * y;
    return t * y + 0.5f;
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SituGluConfigKimi>
inline void calculate_rea_situ_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    if constexpr (is_fp32_dest_acc_en) {
        calculate_situ_glu<is_fp32_dest_acc_en, ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
    } else {
        static_assert(Config::beta_gate == 4.0f, "the sigmoid reuses tanh(gate / 4): beta_gate must be 4");
        constexpr float inv_beta_gate = 1.0f / Config::beta_gate;
        constexpr float inv_beta_up = 1.0f / Config::beta_up;
        // beta_gate * beta_up, applied once to the product of the two tanh values.
        constexpr float beta_prod = Config::beta_gate * Config::beta_up;
        constexpr uint dst_tile_size = 32;
        sfpi::vFloat t1 = TANH_POLY_C1;
        sfpi::vFloat t2 = TANH_POLY_C2;
        sfpi::vFloat t3 = TANH_POLY_C3;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
            sfpi::vFloat tg = _rea_tanh_scaled_(gate, inv_beta_gate, t1, t2, t3);
            sfpi::vFloat situ_a = tg * _rea_sigmoid_from_tanh_quarter_(tg);

            sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
            sfpi::vFloat result = situ_a * _rea_tanh_scaled_(up, inv_beta_up, t1, t2, t3);
            result = result * beta_prod;
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
            sfpi::dst_reg++;
        }
    }
}

inline void rea_situ_glu_init() { situ_glu_init(); }

}  // namespace ckernel::sfpu
