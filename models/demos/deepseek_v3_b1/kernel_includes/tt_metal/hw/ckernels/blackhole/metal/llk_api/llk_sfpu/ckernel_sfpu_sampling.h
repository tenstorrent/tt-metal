// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <bool legacy_compat = true>
inline void calculate_sampling_recip_scalar() {
    sfpi::vFloat in = sfpi::dst_reg[0];
    sfpi::vFloat out;
    if constexpr (legacy_compat) {
        out = ckernel::sfpu::_reciprocal_compat_<APPROX ? 2 : 3>(in);
    } else if constexpr (APPROX) {
        out = ckernel::sfpu::sfpu_reciprocal_iter<0>(in);
    } else if constexpr (DST_ACCUM_MODE) {
        out = ckernel::sfpu::sfpu_reciprocal_iter<2>(in);
    } else {
        out = ckernel::sfpu::sfpu_reciprocal_iter<1>(in);
    }
    if constexpr (!(DST_ACCUM_MODE || APPROX)) {
        out = sfpi::convert<sfpi::vFloat16b>(out, sfpi::RoundMode::Nearest);
    }
    sfpi::dst_reg[0] = out;
}

inline void calculate_sampling_clamp_max_scalar(uint32_t param) {
    const sfpi::vFloat max_val = ckernel::sfpu::Converter::as_float(param);
    sfpi::vFloat in = sfpi::dst_reg[0];
    v_if(in > max_val) { sfpi::dst_reg[0] = max_val; }
    v_endif;
}

template <SfpuType OP>
inline void calculate_sampling_binary_comp_first_column(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    static_assert(
        OP == SfpuType::le || OP == SfpuType::lt || OP == SfpuType::ge,
        "sampling_binary_comp_first_column supports le/lt/ge only");
    constexpr uint dst_tile_size_sfpi = 32;
    constexpr int ITERATIONS_FIRST_COLUMN = 4;

    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::vConst0;

        if constexpr (OP == SfpuType::le) {
            v_if(in0 <= in1) { result = sfpi::vConst1; }
            v_endif;
        } else if constexpr (OP == SfpuType::lt) {
            v_if(in0 < in1) { result = sfpi::vConst1; }
            v_endif;
        } else {
            v_if(in0 >= in1) { result = sfpi::vConst1; }
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg += 2;
    }
}

inline void calculate_sampling_mul_unary_scalar_first_column(uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    constexpr int ITERATIONS_FIRST_COLUMN = 4;

    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val * parameter;
        sfpi::dst_reg += 2;
    }
}

inline void calculate_sampling_add_binary_first_column(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    constexpr uint dst_tile_size_sfpi = 32;
    constexpr int ITERATIONS_FIRST_COLUMN = 4;

    for (int d = 0; d < ITERATIONS_FIRST_COLUMN; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = in0 + in1;
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
