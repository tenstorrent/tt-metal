// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_binary_sfpu_init.h"

/*
 * Binary SFPU invocation helper (Quasar)
 *
 * Mirrors the BH/WH `_sfpu_binary_check_and_call_` API surface so kernels can
 * use the same `SFPU_BINARY_*` macros across architectures. The implementation
 * adapts to Quasar's conventions:
 *
 *   - Quasar's `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>` is a thin
 *     wrapper around `_llk_math_eltwise_sfpu_params_` that ignores its
 *     template `APPROXIMATE` argument and forwards everything verbatim.
 *   - Quasar SFPU binary functors take iterations as a runtime argument; the
 *     existing `llk_math_eltwise_binary_sfpu_binop_*` wrappers always inject
 *     `SFPU_ITERATIONS` before the destination indices. We match that calling
 *     convention here so kernels written against this macro produce the same
 *     functor invocation as the hand-written wrappers.
 *   - Quasar currently supports only `VectorMode::RC`; we assert the same way
 *     `_sfpu_check_and_call_` does in the unary macros header.
 *
 * As with the unary helper, the dst-bound LLK_ASSERTs are intentionally left
 * as TODO -- there is no Quasar equivalent of `get_dest_max_tiles<...>` yet.
 */

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _sfpu_binary_check_and_call_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    [[maybe_unused]] VectorMode vector_mode,
    Args&&... args) {
    LLK_ASSERT(vector_mode == VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_sfpu_params_(
        std::forward<Callable>(sfpu_func),
        0 /*dst_tile_index for addressing*/,
        SFPU_ITERATIONS,
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        std::forward<Args>(args)...);
}

}  // namespace ckernel

/*
 * Helper for variadic-template macros
 *
 * Wrap a comma-separated template-argument list in `(...)` at the call site
 * and strip the outer parens with _SFPU_BIN_EXPAND inside the macro body.
 * The macro is namespaced with the `_BIN_` infix so it cannot collide with
 * the unary-macros header if both are included in the same translation unit.
 */

#define _SFPU_BIN_EXPAND(...) __VA_ARGS__

/*
 * Binary SFPU invocation macros (4 total)
 *
 * All paths funnel through ckernel::_sfpu_binary_check_and_call_<DST_SYNC,
 * DST_ACCUM>(...), which (on Quasar) injects `SFPU_ITERATIONS` and dispatches
 * to `_llk_math_eltwise_sfpu_params_`.
 *
 * Argument order for every call macro is:
 *      (DST_SYNC, DST_ACCUM, FN, TEMPLATES,
 *       DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ... rest ...)
 *
 * NOTE on variadics: kernel code is compiled with -std=c++17, where
 * `__VA_OPT__` is unavailable. We use the GCC `, ##__VA_ARGS__` extension to
 * swallow the preceding comma when no extra args are supplied.
 *
 * NOTE on functor signatures: Quasar binary SFPU functors take iterations as
 * a runtime argument, so functors invoked via these macros must accept
 * `(int iterations, uint32_t dst_in0, uint32_t dst_in1, uint32_t dst_out, ...)`.
 * On BH/WH the corresponding macros do NOT inject iterations -- callers there
 * use functors that take iterations as a template parameter instead.
 */

/*
 * Templated functor in `ckernel::sfpu`, runtime vector_mode expression.
 *   SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                    calculate_sfpu_binary, (APPROX, BinaryOp::MUL, fp32_acc),
 *                    dst_in0, dst_in1, dst_out, vmode);
 */
#define SFPU_BINARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                         \
        ::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

/*
 * Same as SFPU_BINARY_CALL but vector_mode is given as a `VectorMode`
 * enumerator token (e.g. RC, C, R).
 *   SFPU_BINARY_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                         calculate_sfpu_binary, (APPROX, BinaryOp::MUL, fp32_acc),
 *                         RC, dst_in0, dst_in1, dst_out);
 */
#define SFPU_BINARY_CALL_MODE(DST_SYNC, DST_ACCUM, FN, TEMPLATES, MODE, DST_IN0, DST_IN1, DST_OUT, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                       \
        ::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>,                                                \
        DST_IN0,                                                                                        \
        DST_IN1,                                                                                        \
        DST_OUT,                                                                                        \
        ::ckernel::VectorMode::MODE,                                                                    \
        ##__VA_ARGS__)

/*
 * Non-templated functor in `ckernel::sfpu`.
 *   SFPU_BINARY_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                       _calculate_binary_, in0, in1, out, vmode);
 */
#define SFPU_BINARY_CALL_FN(DST_SYNC, DST_ACCUM, FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                 \
        ::ckernel::sfpu::FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

/*
 * Templated functor wrapped in a static_cast for overload disambiguation.
 *   SFPU_BINARY_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
 *                         _binary_fn_,
 *                         (APPROX),
 *                         (void(*)(int, uint32_t, uint32_t, uint32_t)),
 *                         in0, in1, out, vmode);
 */
#define SFPU_BINARY_CALL_CAST(                                                                    \
    DST_SYNC, DST_ACCUM, FN, TEMPLATES, SIGNATURE, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...)   \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                 \
        static_cast<_SFPU_BIN_EXPAND SIGNATURE>(::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>), \
        DST_IN0,                                                                                  \
        DST_IN1,                                                                                  \
        DST_OUT,                                                                                  \
        VECTOR_MODE,                                                                              \
        ##__VA_ARGS__)

/*
 * Binary SFPU init macros (4 total)
 *
 * These mirror the unary/ternary SFPU_INIT* macros and delegate to
 * `ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::OP>` (defined in
 * `llk_math_eltwise_binary_sfpu_init.h`). On Quasar that wrapper is itself
 * a thin shim around `_llk_math_eltwise_sfpu_init_()`, matching what the
 * hand-written `llk_math_eltwise_binary_sfpu_binop_init` wrappers do
 * (the `SfpuType` template parameter is unused on Quasar but kept for API
 * parity with BH/WH).
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
