// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_init.h"

/*
 * Quasar keeps the same macro surface as BH/WH. DST_SYNC and DST_ACCUM are
 * unused until Quasar has an equivalent of get_dest_max_tiles<...>. Binary
 * macro calls are limited to VectorMode::RC for now.
 */

namespace ckernel {

template <DstSync /*DST_SYNC*/, bool /*DST_ACCUM*/>
inline __attribute__((always_inline)) void _sfpu_binary_check_(
    [[maybe_unused]] std::uint32_t dst_index_in0,
    [[maybe_unused]] std::uint32_t dst_index_in1,
    [[maybe_unused]] std::uint32_t dst_index_out,
    VectorMode vector_mode) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
}

}  // namespace ckernel

// Strip the parentheses around the template-argument tuple passed to SFPU_BINARY_CALL.

#define _SFPU_BIN_EXPAND(...) __VA_ARGS__

/*
 * Macro hygiene: DST_IN0/DST_IN1/DST_OUT/VECTOR_MODE are evaluated by both
 * the check and params call. Keep call sites to identifiers/literals, not
 * side effects.
 */
#define SFPU_BINARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE),         \
     _llk_math_eltwise_binary_sfpu_params_(                                                               \
         ::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__))

// Non-templated functor in `ckernel::sfpu`.
#define SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC, DST_ACCUM, FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_binary_check_<DST_SYNC, DST_ACCUM>(DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE),               \
     _llk_math_eltwise_binary_sfpu_params_(                                                                     \
         ::ckernel::sfpu::FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__))

/*
 * Binary SFPU init macros (4 total)
 *
 * These mirror the unary/ternary SFPU_UNARY_INIT* macros and delegate to
 * `ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::OP>` (defined in
 * `llk_math_eltwise_binary_sfpu_init.h`). On Quasar that wrapper is itself
 * a thin shim around `_llk_math_eltwise_sfpu_init_()` that then invokes the
 * init callback (the `SfpuType` template parameter is unused on Quasar but
 * kept for API parity with BH/WH).
 *
 * SfpuType lives inside `::ckernel::` on Quasar (in llk_defs.h), so the
 * fully-qualified path is `::ckernel::SfpuType::OP`.
 */

/*
 * Bare init: no callback.
 *   SFPU_BINARY_INIT(add_fp32);
 */
#define SFPU_BINARY_INIT(OP) ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>()

/*
 * Init with a non-templated callback.
 *   SFPU_BINARY_INIT_FN_NO_ARGS(lcm, sfpu::calculate_sfpu_lcm_init);
 */
#define SFPU_BINARY_INIT_FN_NO_ARGS(OP, INIT_FN) \
    ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN)

/*
 * Init with a templated callback.
 *   SFPU_BINARY_INIT_FN(mul_int32, sfpu::mul_int32_init, (APPROXIMATE));
 */
#define SFPU_BINARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_BIN_EXPAND TEMPLATES>)

/*
 * Init with a templated callback and extra runtime arguments.
 *   SFPU_BINARY_INIT_FN_ARGS(add_fp32, sfpu::add_init, (APPROX), scale);
 */
#define SFPU_BINARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...)              \
    ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>( \
        INIT_FN<_SFPU_BIN_EXPAND TEMPLATES>, ##__VA_ARGS__)
