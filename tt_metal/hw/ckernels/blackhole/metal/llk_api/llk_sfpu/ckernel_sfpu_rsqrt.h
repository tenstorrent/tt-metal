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

}  // namespace sfpu
}  // namespace ckernel

#if !defined(TT_POLY_LLK_DISABLE)
#include "ckernel_sfpu_rsqrt_bf16.h"
#endif

namespace ckernel::sfpu {

#if !defined(TT_POLY_LLK_DISABLE)
template <int ITERATIONS = 8>
inline void calculate_rsqrt_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::calculate_newton_root<ttpoly_generated::RsqrtBf16Config, ITERATIONS>();
}
template <auto...>
inline void init_rsqrt_tt_poly_bf16() {
    ckernel::sfpu::ttpoly::init_newton_root<ttpoly_generated::RsqrtBf16Config>();
}
#endif

}  // namespace ckernel::sfpu
