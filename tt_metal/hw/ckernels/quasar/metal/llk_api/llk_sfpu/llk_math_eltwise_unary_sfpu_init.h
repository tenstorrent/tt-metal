// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu.h"

namespace ckernel {

namespace sfpu {
void softcap_init();
}  // namespace sfpu

// The bare init runs the common SFPU init, plus the op's own init for the ops whose Compute API
// calls the bare form (as on Blackhole).
template <SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_sfpu_init_();
    if constexpr (sfpu_op == SfpuType::softcap) {
        sfpu::softcap_init();
    }
}

// sfpu_op template parameter is unused, but kept for backwards compatibility
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_sfpu_init_();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
