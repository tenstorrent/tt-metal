// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu_init.h"

/*
 * Ternary SFPU invocation helper (Quasar)
 *
 * Mirrors the BH/WH `_sfpu_ternary_check_and_call_` API surface so kernels
 * can use the same `SFPU_TERNARY_*` macros across architectures. The
 * implementation adapts to Quasar's conventions:
 *
 *   - Quasar's `_llk_math_eltwise_ternary_sfpu_params_` takes `int
 *     vector_mode` (not `VectorMode`), so we cast at the boundary.
 *   - Unlike the binary path, Quasar ternary functors take iterations as a
 *     template argument (see `calculate_where<APPROXIMATE, SFPU_ITERATIONS>`
 *     in `llk_math_eltwise_ternary_sfpu_where.h`). We therefore do not
 *     inject `SFPU_ITERATIONS` here -- callers pass it (or any other knob)
 *     through the TEMPLATES argument list, exactly as on BH/WH.
 *   - Quasar currently supports only `VectorMode::RC`; we assert the same
 *     way `_sfpu_check_and_call_` does in the unary/binary macros headers.
 *
 * As with the unary/binary helpers, the dst-bound LLK_ASSERTs are
 * intentionally left as TODO -- there is no Quasar equivalent of
 * `get_dest_max_tiles<...>` yet.
 */

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _sfpu_ternary_check_and_call_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    VectorMode vector_mode,
    Args&&... args) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_ternary_sfpu_params_(
        std::forward<Callable>(sfpu_func),
        dst_index_in0,
        dst_index_in1,
        dst_index_in2,
        dst_index_out,
        vector_mode,
        std::forward<Args>(args)...);
}

}  // namespace ckernel

/*
 * Helper for variadic-template macros
 *
 * Wrap a comma-separated template-argument list in `(...)` at the call site
 * and strip the outer parens with _SFPU_TERN_EXPAND inside the macro body.
 * The macro is namespaced with the `_TERN_` infix so it cannot collide with
 * the unary or binary macros header if all are included in the same TU.
 */

#define _SFPU_TERN_EXPAND(...) __VA_ARGS__

/*
 * Ternary SFPU invocation macros (3 total)
 *
 * All paths funnel through ckernel::_sfpu_ternary_check_and_call_<DST_SYNC,
 * DST_ACCUM>(...), which (on Quasar) asserts RC vector mode and dispatches
 * to `_llk_math_eltwise_ternary_sfpu_params_`.
 *
 * Argument order for every call macro is:
 *      (DST_SYNC, DST_ACCUM, FN, TEMPLATES,
 *       DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ... rest ...)
 *
 * NOTE on variadics: kernel code is compiled with -std=c++17, where
 * `__VA_OPT__` is unavailable. We use the GCC `, ##__VA_ARGS__` extension to
 * swallow the preceding comma when no extra args are supplied.
 */

/*
 * Templated functor in `ckernel::sfpu`, runtime vector_mode expression.
 *   SFPU_TERNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                     calculate_addcmul, (APPROXIMATE, fp32, df, ITER),
 *                     in0, in1, in2, out, vmode, value);
 */
#define SFPU_TERNARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                                  \
        ::ckernel::sfpu::FN<_SFPU_TERN_EXPAND TEMPLATES>,                                                           \
        DST_IN0,                                                                                                    \
        DST_IN1,                                                                                                    \
        DST_IN2,                                                                                                    \
        DST_OUT,                                                                                                    \
        VECTOR_MODE,                                                                                                \
        ##__VA_ARGS__)

/*
 * Non-templated functor in `ckernel::sfpu`.
 *   SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                        _calculate_ternary_, in0, in1, in2, out, vmode);
 */
#define SFPU_TERNARY_CALL_NO_TEMPLATE_ARGS(                                        \
    DST_SYNC, DST_ACCUM, FN, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                 \
        ::ckernel::sfpu::FN, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

/*
 * Templated functor wrapped in a static_cast for overload disambiguation.
 *   SFPU_TERNARY_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                          _ternary_fn_,
 *                          (APPROXIMATE, ITER),
 *                          (void(*)(uint32_t, uint32_t, uint32_t, uint32_t)),
 *                          in0, in1, in2, out, vmode);
 */
#define SFPU_TERNARY_CALL_CAST(                                                                          \
    DST_SYNC, DST_ACCUM, FN, TEMPLATES, SIGNATURE, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                       \
        static_cast<_SFPU_TERN_EXPAND SIGNATURE>(::ckernel::sfpu::FN<_SFPU_TERN_EXPAND TEMPLATES>),      \
        DST_IN0,                                                                                         \
        DST_IN1,                                                                                         \
        DST_IN2,                                                                                         \
        DST_OUT,                                                                                         \
        VECTOR_MODE,                                                                                     \
        ##__VA_ARGS__)

/*
 * Ternary SFPU init macros (4 total)
 *
 * These mirror the unary/binary SFPU_UNARY_INIT* macros and delegate to
 * `ckernel::llk_math_eltwise_ternary_sfpu_init<SfpuType::OP>` (defined in
 * `llk_math_eltwise_ternary_sfpu_init.h`), which on Quasar is a thin
 * inline wrapper around `::_llk_math_eltwise_ternary_sfpu_init_<sfpu_op>()`.
 *
 * SfpuType lives inside `::ckernel::` on Quasar (in llk_defs.h), so the
 * fully-qualified path is `::ckernel::SfpuType::OP`.
 */

/*
 * Bare init: no callback.
 *   SFPU_TERNARY_INIT(where);
 */
#define SFPU_TERNARY_INIT(OP) ::ckernel::llk_math_eltwise_ternary_sfpu_init<::ckernel::SfpuType::OP>()

/*
 * Init with a non-templated callback.
 *   SFPU_TERNARY_INIT_FN_NO_ARGS(some_op, sfpu::some_init);
 */
#define SFPU_TERNARY_INIT_FN_NO_ARGS(OP, INIT_FN) \
    ::ckernel::llk_math_eltwise_ternary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN)

/*
 * Init with a templated callback.
 *   SFPU_TERNARY_INIT_FN(where, sfpu::_init_where_, (APPROXIMATE));
 */
#define SFPU_TERNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_ternary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_TERN_EXPAND TEMPLATES>)

/*
 * Init with a templated callback and extra runtime arguments.
 *   SFPU_TERNARY_INIT_FN_ARGS(some_op, sfpu::some_init, (APPROX), arg0, arg1);
 */
#define SFPU_TERNARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...)              \
    ::ckernel::llk_math_eltwise_ternary_sfpu_init<::ckernel::SfpuType::OP>( \
        INIT_FN<_SFPU_TERN_EXPAND TEMPLATES>, ##__VA_ARGS__)
