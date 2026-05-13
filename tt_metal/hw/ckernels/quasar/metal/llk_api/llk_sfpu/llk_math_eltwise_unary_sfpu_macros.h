// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_common.h"

// =============================================================================
// SFPU invocation helper
//
// TO DO: IMPLEMENT DST_INDEX BOUND CHECK
//
// Validates the destination tile index and dispatches to the LLK unary-SFPU
// params function.
//
// DST_SYNC_MODE and DST_ACCUM_MODE are propagated as template parameters so
// the bound is computed for the kernel's actual sync/accumulation mode.
// All call macros below pass the ambient DST_SYNC_MODE and DST_ACCUM_MODE,
// which every kernel build defines (see jit_build/genfiles.cpp and the
// fake_jit_prelude used by host-side tests).
// =============================================================================

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, typename Callable, typename... Args>
inline __attribute__((always_inline)) void _sfpu_check_and_call_(
    Callable&& sfpu_func, std::uint32_t dst_index, [[maybe_unused]] int vector_mode, Args&&... args) {
    LLK_ASSERT(vector_mode == (int)VectorMode::RC, "Quasar currently only supports vector mode RC");
    _llk_math_eltwise_unary_sfpu_params_(std::forward<Callable>(sfpu_func), dst_index, std::forward<Args>(args)...);
}

}  // namespace ckernel

// =============================================================================
// Helper for variadic-template macros
//
// C preprocessor splits arguments on commas, so `calculate_X<A, B, C>` would
// be seen as three macro arguments. We work around it by wrapping the
// template-argument list in `(...)` at the call site, then stripping the
// outer parentheses with _SFPU_EXPAND inside the macro definition:
//
//   SFPU_CALL((APPROX, ITER), calculate_clamp, dst, vmode, lo, hi);
//                  ^^^^^^^^   <- single macro argument
// =============================================================================

#define _SFPU_EXPAND(...) __VA_ARGS__

// =============================================================================
// SFPU invocation macros (4 total)
//
// All paths funnel through ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>(...),
// which performs the dst-bound LLK_ASSERT and then dispatches to
// _llk_math_eltwise_unary_sfpu_params_.
//
// Argument order for every call macro is:
//      (DST_SYNC, DST_ACCUM, FN, TEMPLATES, ... rest ...)
//
// Making DST_SYNC and DST_ACCUM explicit positional arguments (rather than
// implicitly grabbing the ambient `DST_SYNC_MODE` / `DST_ACCUM_MODE` from
// the surrounding scope) keeps each macro self-contained and lets callers
// in unusual contexts (tests, helper headers, non-standard kernel preludes)
// supply their own values.
//
// NOTE on variadics: kernel code is compiled with -std=c++17, where
// `__VA_OPT__` is unavailable. We use the GCC `, ##__VA_ARGS__` extension to
// swallow the preceding comma when no extra args are supplied; that extension
// is supported by every GCC the kernel toolchain ships with.
// =============================================================================

// Templated functor in `ckernel::sfpu`, runtime vector_mode expression.
//   SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//             calculate_abs,   (APPROXIMATE),                  dst, vmode);
//   SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//             calculate_clamp, (APPROXIMATE, ITER),            dst, vmode, lo, hi);
//   SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//             calculate_celu,  (APPROX, fp32, ITER),           dst, vmode, alpha, alpha_recip);
//   SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE,
//             calculate_rsqrt, (APPROX, 8, fp32, FAST, lgcy),  dst, vmode);
#define SFPU_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>(                       \
        ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__)

// Same as SFPU_CALL but vector_mode is given as a `VectorMode` enumerator name
// (RC, C, RC_custom, ...). The `(int)VectorMode::MODE` cast is generated.
//   SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE,
//                  calculate_erfc,   (8),     RC,        idst);
//   SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE,
//                  calculate_cumsum, (false), RC_custom, dst, first);
#define SFPU_CALL_MODE(DST_SYNC, DST_ACCUM, FN, TEMPLATES, MODE, DST_IDX, ...) \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>(                     \
        ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, (int)::ckernel::VectorMode::MODE, ##__VA_ARGS__)

// Non-templated functor in `ckernel::sfpu`, runtime vector_mode expression.
//   SFPU_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
//                _calculate_top4_,       dst, VectorMode::RC_custom);
//   SFPU_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
//                _calculate_top8_tile_,  dst, VectorMode::RC_custom, tile_index);
//   SFPU_CALL_FN(DST_SYNC_MODE, DST_ACCUM_MODE,
//                calculate_sigmoid_appx, dst, vmode);
#define SFPU_CALL_FN(DST_SYNC, DST_ACCUM, FN, DST_IDX, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__)

// Templated functor wrapped in a static_cast for overload disambiguation.
// SIGNATURE follows TEMPLATES because both relate to FN (template arity then
// the desired pointer signature).
//   SFPU_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
//                  _calculate_threshold_,
//                  (APPROXIMATE, ITER),
//                  (void(*)(uint32_t, uint32_t)),
//                  dst, vmode, p0, p1);
//   SFPU_CALL_CAST(DST_SYNC_MODE, DST_ACCUM_MODE,
//                  _calculate_activation_,
//                  (APPROXIMATE, ACTIVATION, ITER),
//                  (void(*)()),
//                  dst, vmode);
#define SFPU_CALL_CAST(DST_SYNC, DST_ACCUM, FN, TEMPLATES, SIGNATURE, DST_IDX, VECTOR_MODE, ...) \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC, DST_ACCUM>(                                       \
        static_cast<_SFPU_EXPAND SIGNATURE>(::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>),        \
        DST_IDX,                                                                                 \
        VECTOR_MODE,                                                                             \
        ##__VA_ARGS__)

// =============================================================================
// SFPU init macros (3 total)
//
// No dst index involved, so no bound check and no DST_SYNC / DST_ACCUM
// arguments. Argument order mirrors the call macros: OP first (which selects
// the SfpuType), then the init callback (the FN-like argument), then the
// templates that parameterise it.
// =============================================================================

// Bare init: no callback.
//   SFPU_INIT(abs);
#define SFPU_INIT(OP) ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>()

// Init with a templated callback.
//   SFPU_INIT_CB(erf, sfpu::erf_init, (APPROXIMATE));
//   SFPU_INIT_CB(log, sfpu::log_init, (APPROX, fp32, FAST));
#define SFPU_INIT_CB(OP, INIT_CB, TEMPLATES) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_CB<_SFPU_EXPAND TEMPLATES>)

// Init with a templated callback and extra runtime arguments.
//   SFPU_INIT_CB_ARGS(exponential, sfpu::exp_init, (APPROX), scale, clamp_neg);
#define SFPU_INIT_CB_ARGS(OP, INIT_CB, TEMPLATES, ...) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_CB<_SFPU_EXPAND TEMPLATES>, ##__VA_ARGS__)

// ----- complex multi-statement aliases --------------------------------------
//
// These two macros expand to a *sequence* of statements (a static_assert and a
// constexpr definition followed by the SFPU call) rather than a single
// expression, so they cannot be aliased to SFPU_CALL_MODE. We rewrite their
// bodies to call the new helper directly, preserving structure and behavior
// (and gaining the dst-bound assert). They forward the ambient
// DST_SYNC_MODE / DST_ACCUM_MODE explicitly to the helper.

#define SFPU_UNARY_ONE_PARAM_KERNEL_DATA_FORMAT_EXTRA_PARAM(                                      \
    FN, MODE, APPROXIMATE, DATA_FORMAT, EXTRA_PARAM, DST_IDX, PARAM0)                             \
    static_assert(DATA_FORMAT == DataFormat::Int32, "Unsupported data format. Supported: Int32"); \
    constexpr InstrModLoadStore _INSTRUCTION_MODE = InstrModLoadStore::INT32;                     \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC_MODE, DST_ACCUM_MODE>(                              \
        ckernel::sfpu::FN<APPROXIMATE, _INSTRUCTION_MODE, EXTRA_PARAM>, DST_IDX, (int)VectorMode::MODE, PARAM0)

#define SFPU_UNARY_KERNEL_THREE_TEMPLATE_ARGS_FN(FN, APPROXIMATE, DATA_FORMAT, ITERATIONS, DST_IDX, MODE) \
    static_assert(                                                                                        \
        DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||                     \
            DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::Float16,                       \
        "Unsupported data format. Supported data formats are: Float32, Float16_b, Int32, Float16.");      \
    constexpr InstrModLoadStore INSTRUCTION_MODE =                                                        \
        (DATA_FORMAT == DataFormat::Float32 || DATA_FORMAT == DataFormat::Float16_b ||                    \
         DATA_FORMAT == DataFormat::Float16)                                                              \
            ? InstrModLoadStore::DEFAULT                                                                  \
        : (DATA_FORMAT == DataFormat::Int32) ? InstrModLoadStore::INT32                                   \
                                             : InstrModLoadStore::DEFAULT;                                \
    ::ckernel::_sfpu_check_and_call_<DST_SYNC_MODE, DST_ACCUM_MODE>(                                      \
        ckernel::sfpu::FN<APPROXIMATE, INSTRUCTION_MODE, ITERATIONS>, DST_IDX, (int)VectorMode::MODE)
