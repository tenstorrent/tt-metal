// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "cmath_common.h"
#include "ckernel_sfpu_sigmoid.h"

// Clamped SwiGLU (DeepSeek-V4), a fused binary SFPU op:
//
//   result = silu(min(gate, limit)) * clamp(up, -limit, limit)
//
// gate and up are pinned in dst simultaneously, so the activation runs in one pass with no
// intermediate materialized to L1/DRAM. See api/compute/clamped_silu_glu.h for the kernel-facing API.
//
// _sfpu_sigmoid_ takes its reciprocal from sfpu_reciprocal_iter, which requires vConstFloatPrgm0
// to hold 2.0f. Nothing on the binary SFPU init path programs it, and a wrong value yields a
// wrong reciprocal with no fault, so clamped_silu_glu_init below is mandatory.

namespace ckernel::sfpu {

// Distinct from ckernel::sfpu::calculate_swiglu (moe_gpt/device/kernels/swiglu_sfpu.h), the
// gpt-oss / MiniMax-M3 activation reached as RoutedExpertActivation::SwiGluOai: that one shifts
// the up half by +1 and scales the sigmoid argument by alpha=1.702. Here the gate half is a plain
// SiLU, which is what the name says and what a Config of that op could not express -- its `up +
// 1.0f` is hardcoded outside the Config.
//
// calculate_clamped_silu_gate / calculate_clamped in experimental/llk_sfpu/ckernel_sfpu_clamped_silu.h
// are these two halves as unary ops over dst_reg[0]; reusing them costs three passes over dst, not one.
//
// The limit is compile-time so the kernel never loads it from an LReg. V4 Pro and V4 Flash share
// one value (swiglu_limit = 10.0 in the DeepSeek-V4 HF config -- not the gpt-oss swiglu_limit,
// which is 7.0), so one config covers both; other models add a config beside this one.
struct ClampedSiluGluConfigDsV4 {
    static constexpr float limit = 10.0f;
};

// The 16-bit DEST body: the same operations per element as the loop of calculate_clamped_silu_glu, with
// _sfpu_sigmoid_<false> spelled out so the up clamp, the -limit literal and the next row's gate fill its dependency
// stalls. The last row reads one row past the gate tile and drops it.
template <int ITERATIONS, class Config>
sfpi_inline void _calculate_clamped_silu_glu_bf16_(
    const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float limit = Config::limit;
    constexpr float ONE_LN2 = 1.4426950216293334961f;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    sfpi::vFloat gate = sfpi::min(sfpi::dst_reg[gate_tile_idx * dst_tile_size], limit);
    for (int d = 0; d < ITERATIONS; d++) {
        // _sfpu_exp_21f_bf16_<true>(-gate)
        sfpi::vFloat xlog2 = sfpi::clamp(-gate * ONE_LN2 + 127.f, 0.0f, 255.0f);
        sfpi::vFloat z = sfpi::as<sfpi::vFloat>(_float_to_int32_for_exp_21f_(xlog2));
        sfpi::vInt exponential_part = sfpi::exexp(z, sfpi::ExponentMode::Biased);
        sfpi::vFloat frac = sfpi::convert<sfpi::vFloat>(sfpi::exman(z), sfpi::RoundMode::Nearest);
        frac = PolynomialEvaluator::eval(frac, EXP_21F_BF16_C0, EXP_21F_BF16_C1, EXP_21F_BF16_C2);
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat den = 1.0f + sfpi::setexp(frac, exponential_part);

        // sfpu_reciprocal_iter<1>(den)
        sfpi::vFloat lower = -limit;
        sfpi::vFloat sig = sfpi::approx_recip(den);
        sfpi::vFloat t = den * sig - sfpi::vConstFloatPrgm0;
        up = sfpi::max(up, lower);
        v_if(t < 0) { sig = sig * -t - 0.0f; }
        v_endif;

        sfpi::vFloat result = gate * sig;
        sfpi::vFloat gate_next = sfpi::dst_reg[gate_tile_idx * dst_tile_size + 1];
        up = sfpi::min(up, limit);
        result = result * up;
        gate = sfpi::min(gate_next, limit);
        result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = ClampedSiluGluConfigDsV4>
inline void calculate_clamped_silu_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float limit = Config::limit;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

    if constexpr (!is_fp32_dest_acc_en) {
        _calculate_clamped_silu_glu_bf16_<ITERATIONS, Config>(gate_tile_idx, up_tile_idx, out_tile_idx);
        return;
    }

    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];

        // The gate half clamps only the top, the up half clamps both ends. The asymmetry is
        // the model's (DeepseekV4Experts._apply_gate for the routed experts this kernel serves,
        // and DeepseekV4MLP.forward for the dense/shared one), not a saturation bound.
        gate = sfpi::min(gate, limit);
        up = sfpi::clamp(up, -limit, limit);

        // V4's alpha is 1, so the gate half is a plain SiLU over the clamped value.
        sfpi::vFloat result = gate * _sfpu_sigmoid_<is_fp32_dest_acc_en>(gate) * up;
        if constexpr (!is_fp32_dest_acc_en) {
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
        }

        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

inline void clamped_silu_glu_init() {
    // _sfpu_sigmoid_'s own init: it owns the Prgm0 requirement noted above.
    sigmoid_init</*APPROXIMATION_MODE=*/false>();
}

}  // namespace ckernel::sfpu
