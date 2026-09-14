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
// The limit is compile-time so the kernel never loads it from an LReg. V4 Pro and V4 Flash share
// one value (swiglu_limit = 10.0 in the DeepSeek-V4 HF config -- not the gpt-oss swiglu_limit,
// which is 7.0), so one config covers both; other models add a config beside this one.
struct ClampedSiluGluConfigDsV4 {
    static constexpr float limit = 10.0f;
};

template <bool is_fp32_dest_acc_en, int ITERATIONS = 8, class Config = ClampedSiluGluConfigDsV4>
inline void calculate_clamped_silu_glu(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr float limit = Config::limit;
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing

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
