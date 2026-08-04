// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"

/*
 * Quasar ternary SFPU init wrappers
 *
 * Mirrors the BH/WH `ckernel::llk_math_eltwise_ternary_sfpu_init<SfpuType::OP>`
 * API surface so the ternary macros header can share macro definitions across
 * architectures. Both overloads delegate to Quasar's underlying
 * `::_llk_math_eltwise_ternary_sfpu_init_<sfpu_op>()` (declared at file scope
 * in `llk_math_eltwise_ternary_sfpu.h`, made reachable via the `using
 * namespace ckernel;` already present in `llk_math_eltwise_sfpu_common.h`).
 * The `::` prefix is required because we are inside `namespace ckernel` here
 * and the underlying init lives in the global namespace.
 */

namespace ckernel {

template <SfpuType sfpu_op>
inline void llk_math_eltwise_ternary_sfpu_init() {
    ::_llk_math_eltwise_ternary_sfpu_init_<sfpu_op>();
}

template <SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_ternary_sfpu_init(F&& init_func, ARGS&&... args) {
    ::_llk_math_eltwise_ternary_sfpu_init_<sfpu_op>();
    init_func(static_cast<ARGS&&>(args)...);
}

}  // namespace ckernel
