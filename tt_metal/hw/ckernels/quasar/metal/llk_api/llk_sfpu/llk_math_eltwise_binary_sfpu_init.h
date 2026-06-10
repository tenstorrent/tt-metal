// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

/*
 * Quasar binary SFPU init wrappers
 *
 * Mirrors the BH/WH `ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::OP>`
 * API surface so the binary macros header can share macro definitions across
 * architectures. On Quasar there is no dedicated underlying
 * `_llk_math_eltwise_binary_sfpu_init_<sfpu_op>()` helper -- the
 * existing hand-written wrappers (e.g. `llk_math_eltwise_binary_sfpu_binop_init`)
 * just call `_llk_math_eltwise_sfpu_init_()` before invoking an init callback.
 * We do the same here, and accept (but ignore) the `SfpuType` template
 * parameter for API parity with BH/WH and with Quasar's own
 * `llk_math_eltwise_unary_sfpu_init`.
 */

namespace ckernel {

template <[[maybe_unused]] SfpuType sfpu_op>
inline void llk_math_eltwise_binary_sfpu_init() {
    _llk_math_eltwise_sfpu_init_();
}

template <[[maybe_unused]] SfpuType sfpu_op, class F, class... ARGS>
inline void llk_math_eltwise_binary_sfpu_init(F&& init_func, ARGS&&... args) {
    _llk_math_eltwise_sfpu_init_();
    init_func(std::forward<ARGS>(args)...);
}

}  // namespace ckernel
