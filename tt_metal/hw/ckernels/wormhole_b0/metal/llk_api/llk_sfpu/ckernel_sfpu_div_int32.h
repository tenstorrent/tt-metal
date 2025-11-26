// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel::sfpu {
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_div_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

        sfpi::vFloat float_in0 = sfpi::int32_to_float(in0, 0);
        sfpi::vFloat float_in1 = sfpi::int32_to_float(in1, 0);
        sfpi::vFloat result = float_in0 * sfpi::setsgn(_sfpu_reciprocal_<2>(float_in1), float_in1);

        // Try trunc and floor results
        sfpi::vFloat trunc_result = _trunc_body_<APPROXIMATION_MODE, true>(result);
        sfpi::vFloat floor_result = _floor_body_<APPROXIMATION_MODE, true>(result);

        // sfpi::vInt int_result = sfpu::_float_to_int32_(trunc_result);

        // Add check
        sfpi::vFloat prod = result * float_in1;
        sfpi::vInt int_prod = sfpu::_float_to_int32_(prod);
        // A is positive, B is positive --> add 1 to result
        // A is negative, B is negative --> add 1 to result
        // A is positive, B is negative --> sub 1 to result
        // A is negative, B is positive --> sub 1 to result

        // Compare with original numerator (float_in0)
        // sfpi::vMask mask_round_up   = GT(prod, float_in0);      // result*B > A
        // sfpi::vMask mask_round_down = LE(ADD(prod, 1.0f), float_in0); // (result+1)*B <= A

        // Apply corrections
        // int_result = WHERE(mask_round_up, sfpi::sub(int_result, sfpi::int_set1(1)), int_result);
        // int_result = WHERE(mask_round_down, sfpi::add(int_result, sfpi::int_set1(1)), int_result);

        sfpi::vInt int_result = sfpu::_float_to_int32_(result);
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = int_prod;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    _init_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
