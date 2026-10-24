// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// SiLU keeps x live beside the sigmoid, which leaves no lregs for all five of its constants: c1 and c2 come from the
// programmable constants silu_init sets.
struct SiluBf16Constants {
    sfpi::vFloat neg_one_ln2 = -1.4426950216293334961f;
    sfpi::vFloat exp_bias = 127.f;
    sfpi::vFloat c0 = EXP_21F_BF16_C0;
    static constexpr auto& c1 = sfpi::vConstFloatPrgm1;
    static constexpr auto& c2 = sfpi::vConstFloatPrgm2;
};

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en) {
        // The BF16-grade sigmoid with its constants loaded once per call.
        const SiluBf16Constants constants;
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++) {
            sfpi::vFloat x = sfpi::dst_reg[0];
            sfpi::vFloat result = x * _sfpu_sigmoid_bf16_preloaded_(x, constants);
            result = sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest);
            sfpi::dst_reg[0] = result;
            sfpi::dst_reg++;
        }
        return;
    }
    // silu(x) = x * sigmoid(x) with the FP32-accurate sigmoid two rows at a time; x is reloaded from DEST for the
    // product, as the sigmoid leaves no lreg to keep it in.
    const SigmoidFp32Constants constants;
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS / 2; d++) {
        sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
        _sfpu_sigmoid_fp32_rows_(rows, constants);
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::dst_reg[0] = x0 * rows[0];
        sfpi::dst_reg[1] = x1 * rows[1];
        sfpi::dst_reg += 2;
    }
    if constexpr (ITERATIONS % 2 != 0) {
        sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
        _sfpu_sigmoid_fp32_rows_(row, constants);
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * row[0];
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
    sfpi::vConstFloatPrgm1 = EXP_21F_BF16_C1;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2;
}

}  // namespace ckernel::sfpu
