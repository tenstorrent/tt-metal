// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

namespace sfpu {
void isinf_isnan_init();
void softcap_init();
}  // namespace sfpu

// The bare init runs the common SFPU init, plus the op's own init for the ops whose Compute API
// calls the bare form (as on Blackhole).
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_sfpu_init_();
    if constexpr (sfpu_op == SfpuType::softcap) {
        sfpu::softcap_init();
    } else if constexpr (
        sfpu_op == SfpuType::isinf || sfpu_op == SfpuType::isposinf || sfpu_op == SfpuType::isneginf ||
        sfpu_op == SfpuType::isnan || sfpu_op == SfpuType::isfinite) {
        // ADDR_MOD_6 for the store-side Dest increment of _calculate_sfpu_isinf_isnan_.
        sfpu::isinf_isnan_init();
    }
}

// sfpu_op template parameter is unused, but kept for backwards compatibility
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_sfpu_init_();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
