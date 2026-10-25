// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"  // math::reset_counters, p_setrwc
#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

template <bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_silu() {
    if constexpr (!is_fp32_dest_acc_en) {
        // x * sigmoid(x) with the BF16-grade sigmoid two rows at a time, its clamp bound and c2 in the programmable
        // constants silu_init sets; x is reloaded from DEST for the product, as the sigmoid leaves no lreg to keep it.
        const SigmoidBf16Constants constants;
#pragma GCC unroll 4
        for (int d = 0; d < ITERATIONS / 2; d++) {
            sfpi::vFloat rows[2] = {sfpi::dst_reg[0], sfpi::dst_reg[1]};
            _sfpu_sigmoid_bf16_rows_</*constants_in_cregs=*/true>(rows, constants);
            sfpi::vFloat x0 = sfpi::dst_reg[0];
            sfpi::vFloat x1 = sfpi::dst_reg[1];
            sfpi::dst_reg[0] = sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(x0 * rows[0], sfpi::RoundMode::Nearest));
            sfpi::dst_reg[1] = sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(x1 * rows[1], sfpi::RoundMode::Nearest));
            sfpi::dst_reg += 2;
        }
        if constexpr (ITERATIONS % 2 != 0) {
            sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
            _sfpu_sigmoid_bf16_rows_</*constants_in_cregs=*/true>(row, constants);
            sfpi::dst_reg[0] =
                sfpi::vFloat(sfpi::convert<sfpi::vFloat16b>(sfpi::dst_reg[0] * row[0], sfpi::RoundMode::Nearest));
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
        _sfpu_sigmoid_fp32_rows_</*constants_in_cregs=*/false>(rows, constants);
        sfpi::vFloat x0 = sfpi::dst_reg[0];
        sfpi::vFloat x1 = sfpi::dst_reg[1];
        sfpi::dst_reg[0] = x0 * rows[0];
        sfpi::dst_reg[1] = x1 * rows[1];
        sfpi::dst_reg += 2;
    }
    if constexpr (ITERATIONS % 2 != 0) {
        sfpi::vFloat row[1] = {sfpi::dst_reg[0]};
        _sfpu_sigmoid_fp32_rows_</*constants_in_cregs=*/false>(row, constants);
        sfpi::dst_reg[0] = sfpi::dst_reg[0] * row[0];
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void silu_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    // calculate_silu uses the non-approx sigmoid path via _sfpu_sigmoid_, so we must use non-approx sigmoid_init
    sigmoid_init<false>();
    // _sfpu_sigmoid_bf16_rows_<true>'s; the FP32 path loads its constants (_sfpu_sigmoid_fp32_rows_<false>).
    sfpi::vConstFloatPrgm1 = SigmoidBf16Constants::t_max;
    sfpi::vConstFloatPrgm2 = EXP_21F_BF16_C2_UNSCALED;
}

}  // namespace ckernel::sfpu
