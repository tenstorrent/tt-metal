// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_is_fp16_zero.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

inline void mask_init() { math::_reset_counters_<p_setrwc::SET_ABD_F>(); }

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask() {
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask)) { dst_reg[0] = 0.0f; }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_int_mask() {
    const int mask_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vInt mask = dst_reg[mask_idx];
        // Quasar: store as INT32. SFPSTORE's default format follows ALU_FORMAT_SPEC_REG rather than the 32-bit
        // Dest mode; on the Int32 unpack-to-dest path `dst_reg[0] = 0.0f` cleared only the top 16 bits.
        v_if(mask == 0) { dst_reg[0].mode<DataLayout::I32>() = vInt(0); }
        v_endif;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_mask_posinf() {
    const int mask_val_idx = 32;
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat mask = dst_reg[mask_val_idx];
        v_if(_sfpu_is_fp16_zero_(mask)) { dst_reg[0] = std::numeric_limits<float>::infinity(); }
        v_endif;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
