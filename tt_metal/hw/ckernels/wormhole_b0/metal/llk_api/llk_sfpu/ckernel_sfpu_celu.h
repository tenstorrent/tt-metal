// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "sfpu/ckernel_sfpu_exp.h"

namespace ckernel::sfpu {

// CELU: alpha * (exp(x / alpha) - 1)
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, int ITERATIONS>
inline void calculate_celu(std::uint32_t dst_index_in, std::uint32_t dst_index_out, uint32_t param0, uint32_t param1) {
    constexpr std::uint32_t SFP_DST_TILE_ROWS = 32;
    // All params are in FP16_B format
    // param0 = alpha
    // param1 = alpha_recip

    sfpi::vFloat alpha = Converter::as_float(param0);
    sfpi::vFloat alpha_recip = Converter::as_float(param1);
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[dst_index_in * SFP_DST_TILE_ROWS];

        v_if(v < sfpi::vConst0) {
            // Compute exp(x / alpha)
            sfpi::vFloat exp_val =
                _sfpu_exp_21f_bf16_<true>(v * alpha_recip);  // is_fp32_dest_acc_en set to true to avoid rounding as it
                                                             // has to be done at the end of operation

            sfpi::vFloat result = alpha * (exp_val - sfpi::vConst1);
            if constexpr (!is_fp32_dest_acc_en) {
                result = sfpi::reinterpret<sfpi::vFloat>(sfpi::float_to_fp16b(result, sfpi::RoundMode::NearestEven));
            }
            sfpi::dst_reg[dst_index_out * SFP_DST_TILE_ROWS] = result;
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
