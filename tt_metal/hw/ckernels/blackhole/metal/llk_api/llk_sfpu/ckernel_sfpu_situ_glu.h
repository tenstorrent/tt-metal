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
// Identical to sfpu_reciprocal_iter: MAX_ITER == 0 returns the raw estimate and positive values
// apply the same guarded Newton refinements. Keep in sync with recip.h.
template <int MAX_ITER>
sfpi_inline sfpi::vFloat _situ_glu_reciprocal_(const sfpi::vFloat x) {
    sfpi::vFloat y = sfpi::approx_recip(x);
    if constexpr (MAX_ITER == 0) {
        return y;
    }
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

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = SituGluConfigKimi>
inline void calculate_situ_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float beta_gate = Config::beta_gate;
    constexpr float inv_beta_gate = 1.0f / Config::beta_gate;
    constexpr float beta_up = Config::beta_up;
    constexpr float inv_beta_up = 1.0f / Config::beta_up;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        sfpi::vFloat gate_tanh = _sfpu_tanh_polynomial_(gate * inv_beta_gate);
        sfpi::vFloat sigmoid;
        if constexpr (is_fp32_dest_acc_en) {
            sigmoid = _situ_glu_sigmoid_<true>(gate);
        } else {
            // With t=tanh(gate/4), tanh(gate/2)=2t/(1+t^2). Reusing the gate softcap's t
            // removes a third polynomial from sigmoid(x)=0.5*(tanh(x/2)+1).
            sigmoid = 0.5f * (2.0f * gate_tanh * _situ_glu_reciprocal_<0>(1.0f + gate_tanh * gate_tanh) + 1.0f);
        }
        sfpi::vFloat situ_a = beta_gate * gate_tanh * sigmoid;

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
