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
            // Workaround for BH limitations:
            //   • sfpi::int32_to_float cannot correctly convert negative int32 values (see #33044)
            //   • sfpi::setsgn is not functional on BH (see #19675)
            // So we convert inputs to their absolute values, typecast to fp32 and reapply the original signs to the
            // result
            sfpi::vInt pos_in0;
            v_if(in0 < 0) { pos_in0 = 0 - in0; }
            v_else { pos_in0 = in0; }
            v_endif;

            sfpi::vInt pos_in1;
            v_if(in1 < 0) { pos_in1 = 0 - in1; }
            v_else { pos_in1 = in1; }
            v_endif;

            sfpi::vFloat float_in0 = sfpi::int32_to_float(pos_in0, 0);
            sfpi::vFloat float_in1 = sfpi::int32_to_float(pos_in1, 0);
            result = float_in0 * _sfpu_reciprocal_<2>(float_in1);
            result = sfpi::setsgn(result, in0 ^ in1);
        }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void div_init() {
    _init_sfpu_reciprocal_<false>();
}

}  // namespace ckernel::sfpu
