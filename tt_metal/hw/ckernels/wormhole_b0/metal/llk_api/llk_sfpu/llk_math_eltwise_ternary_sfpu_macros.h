// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "llk_math_eltwise_ternary_sfpu_params.h"

// =============================================================================
// Ternary SFPU invocation helper
//
// Validates the four destination tile indices (three inputs, one output) and
// then dispatches to the LLK ternary-SFPU params function. The dst-bound
// checks used to live inside _llk_math_eltwise_ternary_sfpu_params_ itself;
// placing them here keeps the LLK kernel free of host/firmware-side
// preconditions and ensures the assertions are defined exactly once instead
// of being duplicated in every macro.
//
// DST_SYNC_MODE and DST_ACCUM_MODE are propagated as template parameters so
// the bound is computed for the kernel's actual sync/accumulation mode.
// =============================================================================

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _sfpu_ternary_check_and_call_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_in2,
    std::uint32_t dst_index_out,
    int vector_mode,
    Args&&... args) {
    LLK_ASSERT(
        (dst_index_in0 < get_dest_max_tiles<DST_SYNC, DST_ACCUM, DstTileShape::Tile32x32>()),
        "dst_index_in0 exceeds max dest tiles");
    LLK_ASSERT(
        (dst_index_in1 < get_dest_max_tiles<DST_SYNC, DST_ACCUM, DstTileShape::Tile32x32>()),
        "dst_index_in1 exceeds max dest tiles");
    LLK_ASSERT(
        (dst_index_in2 < get_dest_max_tiles<DST_SYNC, DST_ACCUM, DstTileShape::Tile32x32>()),
        "dst_index_in2 exceeds max dest tiles");
    LLK_ASSERT(
        (dst_index_out < get_dest_max_tiles<DST_SYNC, DST_ACCUM, DstTileShape::Tile32x32>()),
        "dst_index_out exceeds max dest tiles");
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

// =============================================================================
// Helper for variadic-template macros
//
// Wrap a comma-separated template-argument list in `(...)` at the call site
// and strip the outer parens with _SFPU_TER_EXPAND inside the macro body.
// The macro is namespaced with the `_TER_` infix so it cannot collide with
// the unary/binary macros headers if multiple are included in the same
// translation unit.
// =============================================================================

#define _SFPU_TER_EXPAND(...) __VA_ARGS__

// =============================================================================
// Ternary SFPU invocation macros (4 total)
//
// All paths funnel through ckernel::_sfpu_ternary_check_and_call_<DST_SYNC,
// DST_ACCUM>(...), which performs the dst-bound LLK_ASSERTs and then
// dispatches to _llk_math_eltwise_ternary_sfpu_params_.
//
// Argument order for every call macro is:
//      (DST_SYNC, DST_ACCUM, FN, TEMPLATES,
//       DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ... rest ...)
//
// NOTE on variadics: kernel code is compiled with -std=c++17, where
// `__VA_OPT__` is unavailable. We use the GCC `, ##__VA_ARGS__` extension to
// swallow the preceding comma when no extra args are supplied.
// =============================================================================

// Templated functor in `ckernel::sfpu`, runtime vector_mode expression.
//   SFPU_TERNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//                     calculate_addcmul, (APPROX, fp32, fmt, ITER),
//                     dst0, dst1, dst2, odst, vmode, value);
#define SFPU_TERNARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                                  \
        ::ckernel::sfpu::FN<_SFPU_TER_EXPAND TEMPLATES>,                                                            \
        DST_IN0,                                                                                                    \
        DST_IN1,                                                                                                    \
        DST_IN2,                                                                                                    \
        DST_OUT,                                                                                                    \
        VECTOR_MODE,                                                                                                \
        ##__VA_ARGS__)

// Same as SFPU_TERNARY_CALL but vector_mode is given as a `VectorMode`
// enumerator name. The `(int)VectorMode::MODE` cast is generated.
//   SFPU_TERNARY_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE,
//                          _calculate_where_, (APPROX, fmt, 8),
//                          RC, dst0, dst1, dst2, odst);
#define SFPU_TERNARY_CALL_MODE(DST_SYNC, DST_ACCUM, FN, TEMPLATES, MODE, DST_IN0, DST_IN1, DST_IN2, DST_OUT, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                                \
        ::ckernel::sfpu::FN<_SFPU_TER_EXPAND TEMPLATES>,                                                          \
        DST_IN0,                                                                                                  \
        DST_IN1,                                                                                                  \
        DST_IN2,                                                                                                  \
        DST_OUT,                                                                                                  \
        (int)::ckernel::VectorMode::MODE,                                                                         \
        ##__VA_ARGS__)

// Non-templated functor in `ckernel::sfpu`.
//   SFPU_TERNARY_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
//                        _some_ternary_fn_, dst0, dst1, dst2, odst, vmode);
#define SFPU_TERNARY_CALL_FN(DST_SYNC, DST_ACCUM, FN, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                          \
        ::ckernel::sfpu::FN, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

// Templated functor wrapped in a static_cast for overload disambiguation.
//   SFPU_TERNARY_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
//                          _some_ternary_fn_,
//                          (APPROX, ITER),
//                          (void(*)(uint32_t, uint32_t, uint32_t, uint32_t)),
//                          dst0, dst1, dst2, odst, vmode);
#define SFPU_TERNARY_CALL_CAST(                                                                          \
    DST_SYNC, DST_ACCUM, FN, TEMPLATES, SIGNATURE, DST_IN0, DST_IN1, DST_IN2, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_ternary_check_and_call_<DST_SYNC, DST_ACCUM>(                                       \
        static_cast<_SFPU_TER_EXPAND SIGNATURE>(::ckernel::sfpu::FN<_SFPU_TER_EXPAND TEMPLATES>),        \
        DST_IN0,                                                                                         \
        DST_IN1,                                                                                         \
        DST_IN2,                                                                                         \
        DST_OUT,                                                                                         \
        VECTOR_MODE,                                                                                     \
        ##__VA_ARGS__)

// =============================================================================
// Ternary SFPU init macros
//
// Unlike the unary/binary paths, ternary has no
// `llk_math_eltwise_ternary_sfpu_init` wrapper (only the raw
// `_llk_math_eltwise_ternary_sfpu_init_<op>()`). We expose only the bare init
// here; callers that need a templated callback can invoke the LLK init and
// the callback explicitly, or add SFPU_TERNARY_INIT_CB if that need arises.
// =============================================================================

// Bare init: no callback.
//   SFPU_TERNARY_INIT(addcmul);
// Note: both `SfpuType` and `_llk_math_eltwise_ternary_sfpu_init_` live in the
// global namespace (the LLK header tt_llk_<arch>/llk_lib/
// llk_math_eltwise_ternary_sfpu.h has no `namespace ckernel { ... }` block,
// unlike the unary/binary init wrappers in tt_metal/hw/ckernels/<arch>/.../
// llk_math_eltwise_*_sfpu_init.h). Both must be qualified with a leading `::`,
// not `::ckernel::`.
#define SFPU_TERNARY_INIT(OP) ::_llk_math_eltwise_ternary_sfpu_init_<::SfpuType::OP>()
