// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_common.h"

namespace ckernel {

// sfpu_op template parameter is unused, but kept for backwards compatibility
template <[[maybe_unused]] SfpuType sfpu_op>
inline void llk_math_eltwise_unary_sfpu_init() {
    _llk_math_eltwise_unary_sfpu_init_();
}

// sfpu_op template parameter is unused, but kept for backwards compatibility
template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_unary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_unary_sfpu_init_();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
