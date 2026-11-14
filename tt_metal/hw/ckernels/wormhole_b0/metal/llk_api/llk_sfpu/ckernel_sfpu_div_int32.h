// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

// Two's complement int32 -> fp32 (round to nearest). The SFPU converts sign-magnitude integers,
// and -2^31 has no sign-magnitude encoding: sfpi::abs(-2^31) stays 0x80000000, which the cast
// reads as -0.0. Patch that lane to 2^31 by its sign bit before re-applying the input's sign.
sfpi_inline sfpi::vFloat int32_to_fp32(sfpi::vInt x) {
    constexpr float INT32_MIN_MAGNITUDE_FP32 = 0x1.0p31f;
    sfpi::vFloat f = sfpi::convert<sfpi::vFloat>(sfpi::abs(x), sfpi::RoundMode::Nearest);
    v_if(f < 0.0f) { f = INT32_MIN_MAGNITUDE_FP32; }
    v_endif;
    return sfpi::copysgn(f, x);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // The inputs are in 2's complement form; read the raw words (I32 layout) so that -2^31 is
        // seen as itself. A sign-magnitude load (vSMag) would turn it into -0 -> +/-0.0 and the
        // quotient into 0 or inf.
        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi].mode<sfpi::DataLayout::I32>();
        // Pure conversions, so they run unconditionally, outside the predicate below.
        sfpi::vFloat float_in0 = int32_to_fp32(in0);
        sfpi::vFloat float_in1 = int32_to_fp32(in1);
        sfpi::vFloat result = 1.0f;

        v_if(in0 == 0 || in0 != in1) {
            sfpi::vFloat recip_in1 = sfpu_reciprocal_iter<2>(float_in1);
            result = float_in0 * recip_in1;
            // Residual correction using the remainder (a - q*b) * (1/b). The reciprocal is only
            // accurate to ~1 ulp, so exact quotients (e.g. 28/14) can land 1 ulp low (1.9999998).
            // One residual step snaps exact quotients to the correct value.
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
