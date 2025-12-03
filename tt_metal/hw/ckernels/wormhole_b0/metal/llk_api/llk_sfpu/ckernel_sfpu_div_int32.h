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
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vInt in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = 0.0f;

        v_if(in0 != 0 && in0 == in1) { result = sfpi::vConst1; }
        v_else {
            sfpi::vFloat float_in0 = sfpi::int32_to_float(in0, 0);
            sfpi::vFloat float_in1 = sfpi::int32_to_float(in1, 0);
            result = float_in0 * _sfpu_reciprocal_<2>(float_in1);
        }
        v_endif;

        // If the result is expected to be integer, this function can be used.
        // vInt int_result = sfpu::_float_to_int32_(result);

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
