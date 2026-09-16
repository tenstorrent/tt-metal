// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu_types.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpu_san.h"
#include "sanitizer/api.h"

namespace ckernel {

template <SfpuType sfpu_op>
inline void llk_math_eltwise_binary_sfpu_init() {
    if constexpr (!sfpu_operation<sfpu_op>::modelled) {
        SAN_HOOK(unsupported());
    }
    _llk_math_eltwise_binary_sfpu_init_<sfpu_op>();
}

template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_binary_sfpu_init(F&& init_func, ARGS&&... args) {
    if constexpr (!sfpu_operation<sfpu_op>::modelled) {
        SAN_HOOK(unsupported());
    }
    _llk_math_eltwise_binary_sfpu_init_<sfpu_op>();
    init_func(static_cast<ARGS&&>(args)...);
}

}  // namespace ckernel
