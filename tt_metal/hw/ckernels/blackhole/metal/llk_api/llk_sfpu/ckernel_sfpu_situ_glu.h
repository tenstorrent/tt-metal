// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_softcap.h"
#include "ckernel_sfpu_tanh.h"

// SiTU-GLU activation, a fused binary SFPU op:
//
//   situ_a  = beta_gate * tanh(gate / beta_gate) * sigmoid(gate)
//   up_half = beta_up   * tanh(up   / beta_up)
//   result  = situ_a * up_half
//
// gate and up are pinned in dst simultaneously, so the activation runs in one pass with no
// intermediate materialized to L1/DRAM. See api/compute/situ_glu.h for the kernel-facing API.
//
// Init: tanh_init claims all three vConstFloatPrgm registers, so the sigmoid half cannot use
// the stock reciprocal -- see _situ_glu_reciprocal_ below.

namespace ckernel::sfpu {

// Betas are compile-time so the kernel never divides and no LReg holds them.
struct SituGluConfigKimi {
    static constexpr float beta_gate = 4.0f;
    static constexpr float beta_up = 25.0f;
};

// Newton reciprocal with 2.0 as a literal. The stock sfpu_reciprocal_iter reads that
// constant from vConstFloatPrgm0, which tanh_init has loaded with a tanh coefficient.
// Identical to sfpu_reciprocal_iter for MAX_ITER >= 1, the only values instantiated here;
// it drops the stock MAX_ITER == 0 case, which returns the raw estimate. Keep in sync with recip.h.
template <int MAX_ITER>
sfpi_inline sfpi::vFloat _situ_glu_reciprocal_(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    // t is negated so NaN detection is a sign check (comparisons against NaN are all
    // false, keeping the correct seed for x=0/inf). `- 0.0f` is SFPMAD shape and
    // preserves signed zero.
    sfpi::vFloat t = x * y - 2.0f;
    if constexpr (MAX_ITER > 1) {
        sfpi::vFloat y1 = y * -t - 0.0f;
        v_if(t < 0) {
            t = x * y1 - 2.0f;
            y = y1 * -t - 0.0f;
        }
        v_endif;
    } else {
        v_if(t < 0) { y = y * -t - 0.0f; }
        v_endif;
    }
    return y;
}

// sigmoid(x) = 1 / (1 + exp(-x)); both exp variants are free of vConstFloatPrgm.
template <bool is_fp32_dest_acc_en>
sfpi_inline sfpi::vFloat _situ_glu_sigmoid_(sfpi::vFloat x) {
    sfpi::vFloat exp_neg_x;
    if constexpr (is_fp32_dest_acc_en) {
        exp_neg_x = _sfpu_exp_accurate_<true>(-x);
    } else {
        exp_neg_x = _sfpu_exp_21f_bf16_<true>(-x);
    }
    return _situ_glu_reciprocal_<is_fp32_dest_acc_en ? 2 : 1>(1.0f + exp_neg_x);
}

// The tanh polynomial of _sfpu_softcap_ for two vectors in lockstep, so each Horner chain fills the other's SFPMAD
// stalls and they share the coefficient loads; the copysgn and the beta multiply are left to the caller.
sfpi_inline void _situ_glu_tanh_x2_(sfpi::vFloat& tg, sfpi::vFloat& tu, sfpi::vFloat xg, sfpi::vFloat xu) {
    sfpi::vFloat ag = sfpi::abs(xg);
    sfpi::vFloat au = sfpi::abs(xu);
    tg = sfpi::vConstFloatPrgm1 + ag * sfpi::vConstFloatPrgm0;
    tu = sfpi::vConstFloatPrgm1 + au * sfpi::vConstFloatPrgm0;
    tg = sfpi::vConstFloatPrgm2 + ag * tg;
    tu = sfpi::vConstFloatPrgm2 + au * tu;
    sfpi::vFloat c = TANH_POLY_C3;
    tg = c + ag * tg;
    tu = c + au * tu;
    c = TANH_POLY_C2;
    tg = c + ag * tg;
    tu = c + au * tu;
    c = TANH_POLY_C1;
    tg = c + ag * tg;
    tu = c + au * tu;
    tg = 0.0f + ag * tg;
    tu = 0.0f + au * tu;
    tg = sfpi::min(tg, 1.0f);
    tu = sfpi::min(tu, 1.0f);
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SituGluConfigKimi>
inline void calculate_situ_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float beta_gate = Config::beta_gate;
    constexpr float inv_beta_gate = 1.0f / Config::beta_gate;
    constexpr float beta_up = Config::beta_up;
    constexpr float inv_beta_up = 1.0f / Config::beta_up;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    if constexpr (!is_fp32_dest_acc_en) {
        constexpr float ONE_LN2 = 1.4426950216293334961f;
        // The next row's gate and 1 / beta_up are loaded into the stalls of this row's two final multiplies; the
        // last row reads one row past the gate tile and drops it.
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat inv_up = inv_beta_up;
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
            sfpi::vFloat xg = gate * inv_beta_gate;
            sfpi::vFloat xu = up * inv_up;
            sfpi::vFloat tg, tu;
            _situ_glu_tanh_x2_(tg, tu, xg, xu);
            sfpi::vFloat softcap_up = sfpi::copysgn(tu, xu) * beta_up;

            // _situ_glu_sigmoid_<false>(gate) spelled out, with the 2.0 literal and the gate softcap's last two
            // steps in its dependency stalls.
            sfpi::vFloat xlog2 = sfpi::clamp(-gate * ONE_LN2 + 127.f, 0.0f, 255.0f);
            sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
            sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
            sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
            frac = PolynomialEvaluator::eval(frac, EXP_21F_BF16_C0, EXP_21F_BF16_C1, EXP_21F_BF16_C2);
            sfpi::vFloat two = 2.0f;
            sfpi::vFloat den = 1.0f + sfpi::setexp(frac, exponential_part);
            sfpi::vFloat softcap_gate = sfpi::copysgn(tg, xg);
            sfpi::vFloat sig = sfpi::approx_recip(den);
            sfpi::vFloat t = den * sig - two;
            softcap_gate = softcap_gate * beta_gate;
            v_if(t < 0) { sig = sig * -t - 0.0f; }
            v_endif;

            sfpi::vFloat situ_a = softcap_gate * sig;
            gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size + 1];
            sfpi::vFloat result = situ_a * softcap_up;
            inv_up = inv_beta_up;
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

            sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
            sfpi::dst_reg++;
        }
        return;
    }

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        // sigmoid takes the raw gate, not the capped value.
        sfpi::vFloat situ_a =
            _sfpu_softcap_(gate, beta_gate, inv_beta_gate) * _situ_glu_sigmoid_<is_fp32_dest_acc_en>(gate);

        sfpi::vFloat result = situ_a * _sfpu_softcap_(up, beta_up, inv_beta_up);
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

inline void situ_glu_init() {
    // One tanh init serves the whole op; the sigmoid half claims no vConstFloatPrgm.
    tanh_init</*APPROXIMATION_MODE=*/false, /*is_fp32_dest_acc_en=*/false>();
}

}  // namespace ckernel::sfpu
