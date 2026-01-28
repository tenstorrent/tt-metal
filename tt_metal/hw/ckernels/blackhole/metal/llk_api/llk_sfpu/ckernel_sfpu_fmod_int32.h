// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_remainder_int32.h"
#include "sfpi.h"

namespace ckernel::sfpu {

// FMOD = a - trunc(a / b) * b
// Implemented using 32-bit integer remainder kernel (see ckernel_sfpu_remainder_int32.h)
sfpi_inline void calculate_fmod_int32_body(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
    constexpr uint dst_tile_size_sfpi = 32;

    // Read inputs
    sfpi::vInt a_signed = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
    sfpi::vInt b_signed = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];

    // Compute unsigned remainder
    sfpi::vInt r = compute_unsigned_remainder_int32(a_signed, b_signed);

    // FMOD sign handling (result has the same sign as a)
    v_if(a_signed < 0) { r = -r; }
    v_endif;

    sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = r;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_fmod_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        calculate_fmod_int32_body(dst_index_in0, dst_index_in1, dst_index_out);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
inline void fmod_int32_init() {
    div_floor_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
