// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"

// =============================================================================
// Binary SFPU invocation helper
//
// Validates the three destination tile indices (two inputs, one output) and
// then dispatches to the LLK binary-SFPU params function. The dst-bound check
// used to live inside _llk_math_eltwise_binary_sfpu_params_ itself; placing
// it here keeps the LLK kernel free of host/firmware-side preconditions and
// ensures the assertions are defined exactly once instead of being duplicated
// in every macro.
//
// DST_SYNC_MODE and DST_ACCUM_MODE are propagated as template parameters so
// the bound is computed for the kernel's actual sync/accumulation mode.
// =============================================================================

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _sfpu_binary_check_and_call_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
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
        (dst_index_out < get_dest_max_tiles<DST_SYNC, DST_ACCUM, DstTileShape::Tile32x32>()),
        "dst_index_out exceeds max dest tiles");
    _llk_math_eltwise_binary_sfpu_params_(
        std::forward<Callable>(sfpu_func),
        dst_index_in0,
        dst_index_in1,
        dst_index_out,
        vector_mode,
        std::forward<Args>(args)...);
}

}  // namespace ckernel

// =============================================================================
// Helper for variadic-template macros
//
// Wrap a comma-separated template-argument list in `(...)` at the call site
// and strip the outer parens with _SFPU_BIN_EXPAND inside the macro body.
// The macro is namespaced with the `_BIN_` infix so it cannot collide with
// the unary-macros header if both are included in the same translation unit.
// =============================================================================

#define _SFPU_BIN_EXPAND(...) __VA_ARGS__

// =============================================================================
// Binary SFPU invocation macros (4 total)
//
// All paths funnel through ckernel::_sfpu_binary_check_and_call_<DST_SYNC,
// DST_ACCUM>(...), which performs the dst-bound LLK_ASSERTs and then
// dispatches to _llk_math_eltwise_binary_sfpu_params_.
//
// Argument order for every call macro is:
//      (DST_SYNC, DST_ACCUM, FN, TEMPLATES,
//       DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ... rest ...)
//
// NOTE on variadics: kernel code is compiled with -std=c++17, where
// `__VA_OPT__` is unavailable. We use the GCC `, ##__VA_ARGS__` extension to
// swallow the preceding comma when no extra args are supplied.
// =============================================================================

// Templated functor in `ckernel::sfpu`, runtime vector_mode expression.
//   SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//                    calculate_add_fp32, (APPROXIMATE),
//                    dst_in0, dst_in1, dst_out, vmode);
#define SFPU_BINARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                         \
        ::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

// Same as SFPU_BINARY_CALL but vector_mode is given as a `VectorMode`
// enumerator name. The `(int)VectorMode::MODE` cast is generated.
//   SFPU_BINARY_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE,
//                         calculate_add_fp32, (APPROXIMATE),
//                         RC, dst_in0, dst_in1, dst_out);
#define SFPU_BINARY_CALL_MODE(DST_SYNC, DST_ACCUM, FN, TEMPLATES, MODE, DST_IN0, DST_IN1, DST_OUT, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                       \
        ::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>,                                                \
        DST_IN0,                                                                                        \
        DST_IN1,                                                                                        \
        DST_OUT,                                                                                        \
        (int)::ckernel::VectorMode::MODE,                                                               \
        ##__VA_ARGS__)

// Non-templated functor in `ckernel::sfpu`.
//   SFPU_BINARY_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
//                       _calculate_binary_, in0, in1, out, vmode);
#define SFPU_BINARY_CALL_FN(DST_SYNC, DST_ACCUM, FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                 \
        ::ckernel::sfpu::FN, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ##__VA_ARGS__)

// Templated functor wrapped in a static_cast for overload disambiguation.
//   SFPU_BINARY_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
//                         _binary_fn_,
//                         (APPROXIMATE, ITER),
//                         (void(*)(uint32_t, uint32_t, uint32_t)),
//                         in0, in1, out, vmode);
#define SFPU_BINARY_CALL_CAST(                                                                    \
    DST_SYNC, DST_ACCUM, FN, TEMPLATES, SIGNATURE, DST_IN0, DST_IN1, DST_OUT, VECTOR_MODE, ...)   \
    ::ckernel::_sfpu_binary_check_and_call_<DST_SYNC, DST_ACCUM>(                                 \
        static_cast<_SFPU_BIN_EXPAND SIGNATURE>(::ckernel::sfpu::FN<_SFPU_BIN_EXPAND TEMPLATES>), \
        DST_IN0,                                                                                  \
        DST_IN1,                                                                                  \
        DST_OUT,                                                                                  \
        VECTOR_MODE,                                                                              \
        ##__VA_ARGS__)

// =============================================================================
// Binary SFPU init macros (3 total)
//
// These mirror the unary SFPU_INIT* macros and delegate to the existing
// `ckernel::llk_math_eltwise_binary_sfpu_init<SfpuType::OP>` wrapper, which
// configures the address-modifier registers for binary SFPU ops and then
// invokes the optional per-op init callback.
// =============================================================================

// Bare init: no callback.
//   SFPU_BINARY_INIT(add_fp32);
#define SFPU_BINARY_INIT(OP) ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>()

// Init with a templated callback.
//   SFPU_BINARY_INIT_CB(mul_int32, sfpu::mul_int32_init, (APPROXIMATE));
#define SFPU_BINARY_INIT_CB(OP, INIT_CB, TEMPLATES) \
    ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>(INIT_CB<_SFPU_BIN_EXPAND TEMPLATES>)

// Init with a templated callback and extra runtime arguments.
//   SFPU_BINARY_INIT_CB_ARGS(add_fp32, sfpu::add_init, (APPROX), scale);
#define SFPU_BINARY_INIT_CB_ARGS(OP, INIT_CB, TEMPLATES, ...)              \
    ::ckernel::llk_math_eltwise_binary_sfpu_init<::ckernel::SfpuType::OP>( \
        INIT_CB<_SFPU_BIN_EXPAND TEMPLATES>, ##__VA_ARGS__)
