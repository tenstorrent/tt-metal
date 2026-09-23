// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_sqrt.h"
#include "cmath_common.h"
#include "sfpu/ckernel_sfpu_rsqrt_compat.h"
#include "sfpi.h"
#include "llk_math_eltwise_sfpu_op.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool fp32_dest_acc_en, bool FAST_APPROX, bool legacy_compat>
inline void calculate_rsqrt() {
    if constexpr (legacy_compat) {
        _calculate_rsqrt_compat_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en>(ITERATIONS);
    } else {
        _calculate_sqrt_internal_<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, true, FAST_APPROX>();
    }
}

template <bool APPROXIMATION_MODE, bool legacy_compat>
void rsqrt_init() {
    math::reset_counters(p_setrwc::SET_ABD_F);
    if constexpr (!legacy_compat) {
        sqrt_init<APPROXIMATION_MODE>();
    }
}

// ---------------------------------------------------------------------------------------------------
// Rsqrt<APPROX, FAST_APPROX, LEGACY_COMPAT, DST_SYNC, DST_ACCUM, ITERATIONS>
//   calculate(dst_index, vector_mode) -> calculate_rsqrt
//   init()                            -> rsqrt_init
// Backs rsqrt_tile / rsqrt_tile_init.
// ---------------------------------------------------------------------------------------------------
template <
    bool APPROXIMATION_MODE,
    bool FAST_APPROX,
    bool LEGACY_COMPAT,
    DstSync DST_SYNC,
    bool DST_ACCUM,
    int ITERATIONS = 8>
struct Rsqrt : SfpuUnaryOp<
                   Rsqrt<APPROXIMATION_MODE, FAST_APPROX, LEGACY_COMPAT, DST_SYNC, DST_ACCUM, ITERATIONS>,
                   DST_SYNC,
                   DST_ACCUM> {
    static void kernel() { calculate_rsqrt<APPROXIMATION_MODE, ITERATIONS, DST_ACCUM, FAST_APPROX, LEGACY_COMPAT>(); }

    static void init_kernel() { rsqrt_init<APPROXIMATION_MODE, LEGACY_COMPAT>(); }
};
}  // namespace sfpu
}  // namespace ckernel
