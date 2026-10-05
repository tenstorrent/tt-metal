// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sqrt.h"
#include "cmath_common.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

bool bf16_dest_rsqrt();
template <int ITERATIONS>
void calculate_rsqrt_bf16();
void init_rsqrt_bf16();
// Whether BF16 DEST runs the generated rsqrt kernel as one call over the whole tile.
inline constexpr bool rsqrt_bf16_whole_tile = true;

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_rsqrt() {
    if constexpr (!fp32_dest_acc_en && !FAST_APPROX && !APPROXIMATION_MODE && ITERATIONS == 32) {
        if (bf16_dest_rsqrt()) {
            init_rsqrt_bf16();
            calculate_rsqrt_bf16<ITERATIONS>();
            return;
        }
    }
    _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>();
}

template <bool APPROXIMATION_MODE>
void rsqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace sfpu
}  // namespace ckernel

#include "ckernel_sfpu_rsqrt_bf16.h"
