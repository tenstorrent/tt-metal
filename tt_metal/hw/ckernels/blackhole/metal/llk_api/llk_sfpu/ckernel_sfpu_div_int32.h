// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // The inputs are in 2's complement form
        sfpi::vSMag in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vSMag in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vFloat result = 1.0f;

        v_if(in0 == 0 || in0 != in1) {
            sfpi::vFloat float_in0 = sfpi::convert<sfpi::vFloat>(in0, sfpi::RoundMode::Nearest);
            sfpi::vFloat float_in1 = sfpi::convert<sfpi::vFloat>(in1, sfpi::RoundMode::Nearest);
            sfpi::vFloat recip_in1 = sfpu_reciprocal_iter<2>(float_in1);
            result = float_in0 * recip_in1;
            // Residual correction using the remainder (a - q*b) * (1/b). The reciprocal is only
            // accurate to ~1 ulp, so exact quotients (e.g. 28/14) can land 1 ulp off (2.0000001).
            // One residual step snaps exact quotients to the correct value
            // Note: this cannot recover precision already lost when |operand| > 2^24 is rounded during
            // the int32 -> fp32 conversion.
            //   in1 == 0: recip is +/-inf and (a - inf*0) would produce a NaN, corrupting the
            //             intended inf/-inf/NaN result of division by zero.
            v_if(in1 != 0) { result = result + (float_in0 - result * float_in1) * recip_in1; }
            v_endif;
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    sfpu_reciprocal_init<false>();
}

}  // namespace ckernel::sfpu
