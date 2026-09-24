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
#include "llk_math_eltwise_unary_sfpu_params.h"

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

// Op class for the reciprocal square root.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    bool fp32_dest_acc_en = false,
    bool FAST_APPROX = false,
    bool legacy_compat = false>
struct Rsqrt : SfpuUnaryOp<Rsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, legacy_compat>> {
    static constexpr auto& calculate =
        calculate_rsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, legacy_compat>;
    static inline __attribute__((always_inline)) void init_op() { rsqrt_init<APPROXIMATION_MODE, legacy_compat>(); }
};

}  // namespace sfpu
}  // namespace ckernel
