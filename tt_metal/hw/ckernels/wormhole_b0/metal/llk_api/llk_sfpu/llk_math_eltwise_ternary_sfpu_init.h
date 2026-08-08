// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_ternary_sfpu.h"

namespace ckernel {

template <SfpuType sfpu_op>
inline void llk_math_eltwise_ternary_sfpu_init() {
    _llk_math_eltwise_ternary_sfpu_init_<sfpu_op>();
}

template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_ternary_sfpu_init(F&& init_func, ARGS&&... args) {
    // snake_beta runs as a single self-contained sfpu::snake_beta_init() (no generic prefix) for the sanitizer's
    // per-op operation_init hook; its init_func already inlines the common init, and it uses the scalar
    // reciprocal path (no SFPLOADMACRO), so there is no shared-macro race and no mutex is needed. Other ternary
    // ops keep the prefix + init_func split.
    if constexpr (sfpu_op != SfpuType::snake_beta) {
        _llk_math_eltwise_ternary_sfpu_init_<sfpu_op>();
    }
    init_func(static_cast<ARGS&&>(args)...);
}

}  // namespace ckernel
