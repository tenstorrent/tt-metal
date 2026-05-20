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
// Validates the destination tile index and then dispatches to the LLK SFPU
// params function. The dst-bound check used to live inside
// _llk_math_eltwise_unary_sfpu_params_ itself; placing it here keeps the LLK
// kernel free of host/firmware-side preconditions and ensures the assertion
// is defined exactly once instead of being duplicated in every macro.
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

// =============================================================================
// Backward-compatible aliases
//
// All previous macros are preserved as thin wrappers around the new helper /
// macros so that existing call sites continue to work unchanged AND
// automatically pick up the dst-bound LLK_ASSERT. Each alias forwards the
// ambient `DST_SYNC_MODE` and `DST_ACCUM_MODE` symbols (defined by the kernel
// build prelude) into the new explicit-DST-mode positional slots.
//
// Prefer the SFPU_CALL / SFPU_CALL_MODE / SFPU_CALL_FN / SFPU_CALL_CAST /
// SFPU_INIT* macros above for new code; the names below may be removed once
// every consumer has been migrated.
// =============================================================================

// ----- init aliases ----------------------------------------------------------

#define SFPU_UNARY_KERNEL_INIT(OP, APPROXIMATE) SFPU_INIT(OP)

#define SFPU_INIT_KERNEL_CALL(OP, INIT_CB, APPROXIMATE) SFPU_INIT_CB(OP, INIT_CB, (APPROXIMATE))

#define SFPU_ONE_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, PARAM0) \
    SFPU_INIT_CB_ARGS(OP, INIT_CB, (APPROXIMATE), PARAM0)

#define SFPU_TWO_TEMPLATE_PARAM_INIT(OP, INIT_CB, APPROXIMATE, PARAM0) SFPU_INIT_CB(OP, INIT_CB, (APPROXIMATE, PARAM0))

#define SFPU_THREE_TEMPLATE_PARAM_INIT(OP, INIT_CB, APPROXIMATE, PARAM0, PARAM1) \
    SFPU_INIT_CB(OP, INIT_CB, (APPROXIMATE, PARAM0, PARAM1))

#define SFPU_TWO_PARAM_KERNEL_INIT(OP, INIT_CB, APPROXIMATE, PARAM0, PARAM1) \
    SFPU_INIT_CB_ARGS(OP, INIT_CB, (APPROXIMATE), PARAM0, PARAM1)

#define SFPU_TEMPLATE_INIT_KERNEL(OP, INIT_CB, APPROX, SCALE, CLAMP_NEGATIVE) \
    SFPU_INIT_CB(OP, INIT_CB, (APPROX, SCALE, CLAMP_NEGATIVE))

// ----- compare-with-zero aliases --------------------------------------------

#define SFPU_COMP_INT32_KERNEL(OP, MODE, APPROXIMATE, ITERATIONS, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(                                                                \
        DST_SYNC_MODE,                                                             \
        DST_ACCUM_MODE,                                                            \
        calculate_comp_unary_int,                                                  \
        (APPROXIMATE, SfpuType::OP, ITERATIONS),                                   \
        MODE,                                                                      \
        DST_IDX,                                                                   \
        PARAM0)

#define SFPU_COMP_INT32_KERNEL_UNDERSCORE(OP, MODE, APPROXIMATE, ITERATIONS, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(                                                                           \
        DST_SYNC_MODE,                                                                        \
        DST_ACCUM_MODE,                                                                       \
        _calculate_comp_unary_int_,                                                           \
        (APPROXIMATE, SfpuType::OP, ITERATIONS),                                              \
        MODE,                                                                                 \
        DST_IDX,                                                                              \
        PARAM0)

#define SFPU_ZERO_KERNEL(OP, MODE, APPROXIMATE, DST_IDX) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_comp, (APPROXIMATE, SfpuType::OP), MODE, DST_IDX, 8)

#define SFPU_ZERO_KERNEL_TYPE(TYPE, OP, MODE, APPROXIMATE, DST_IDX) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, TYPE, (APPROXIMATE, SfpuType::OP), MODE, DST_IDX)

// ----- literal-mode invocation aliases --------------------------------------

#define SFPU_UNARY_NO_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), MODE, DST_IDX)

#define SFPU_UNARY_NO_PARAM_KERNEL_FN_ITERATIONS(FN, MODE, APPROXIMATE, DST_IDX, ITERATIONS) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, ITERATIONS), MODE, DST_IDX)

#define SFPU_UNARY_NO_PARAM_KERNEL_WITH_TYPE_AND_ITERATIONS(FN, TYPE, MODE, APPROXIMATE, DST_IDX, ITERATIONS) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (SfpuType::TYPE, APPROXIMATE, ITERATIONS), MODE, DST_IDX)

#define SFPU_UNARY_ONE_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), MODE, DST_IDX, PARAM0)

#define SFPU_UNARY_ONE_PARAM_KERNEL_EXTRA_PARAM(FN, MODE, APPROXIMATE, EXTRA_PARAM, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, EXTRA_PARAM), MODE, DST_IDX, PARAM0)

#define SFPU_UNARY_TWO_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), MODE, DST_IDX, PARAM0, PARAM1)

#define SFPU_UNARY_TWO_PARAM_KERNEL_WITH_DST_ACCUM(FN, MODE, APPROXIMATE, DST_ACCUM, DST_IDX, PARAM0, PARAM1) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, DST_ACCUM), MODE, DST_IDX, PARAM0, PARAM1)

#define SFPU_UNARY_THREE_PARAM_KERNEL_FN(FN, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1, PARAM2) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), MODE, DST_IDX, PARAM0, PARAM1, PARAM2)

#define SFPU_UNARY_THREE_PARAM_KERNEL_WITH_DST_ACCUM(                  \
    FN, MODE, APPROXIMATE, DST_ACCUM, DST_IDX, PARAM0, PARAM1, PARAM2) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, DST_ACCUM), MODE, DST_IDX, PARAM0, PARAM1, PARAM2)

#define SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS(FN, MODE, APPROXIMATE, DST_IDX, PARAM0, PARAM1) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), MODE, DST_IDX, PARAM0, PARAM1)

#define SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (SFPU_ITERATIONS), MODE, DST_IDX, PARAM0)

#define SFPU_UNARY_ONE_PARAM_KERNEL_FN_INT(FN, MODE, APPROXIMATE, DST_IDX, PARAM0) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (uint32_t, APPROXIMATE, 8, uint32_t), MODE, DST_IDX, PARAM0)

#define SFPU_UNARY_NO_PARAM_KERNEL_LOG1P_FN(FN, MODE, APPROXIMATE, FAST_APPROX, FP32, DST_IDX) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, FAST_APPROX, FP32), MODE, DST_IDX)

#define SFPU_FOUR_PARAM_KERNEL_ITER_FIRST_FN(FN, APPROXIMATE, ITER, FP32, LEGACY_COMPAT, DST_IDX, MODE) \
    SFPU_CALL_MODE(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, ITER, FP32, LEGACY_COMPAT), MODE, DST_IDX)

#define SFPU_FIVE_PARAM_KERNEL_ITER_FIRST_FN(FN, APPROXIMATE, ITER, FP32, FAST_APPROX, LEGACY_COMPAT, DST_IDX, MODE) \
    SFPU_CALL_MODE(                                                                                                  \
        DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, ITER, FP32, FAST_APPROX, LEGACY_COMPAT), MODE, DST_IDX)

// SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN takes its mode as a runtime expression,
// not as a VectorMode enumerator name (note the lack of `(int)VectorMode::` in
// the original definition), so it maps to SFPU_CALL rather than SFPU_CALL_MODE.
#define SFPU_FOUR_PARAM_KERNEL_FP32_FIRST_FN(FN, APPROXIMATE, FP32, ITER, LEGACY_COMPAT, DST_IDX, MODE) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, FP32, ITER, LEGACY_COMPAT), DST_IDX, MODE)

// SFPU_TEMPLATE_PARAMS_KERNEL_FN already takes vector_mode as a runtime
// expression named VECTOR_MODE.
#define SFPU_TEMPLATE_PARAMS_KERNEL_FN(                                                                      \
    FN, APPROXIMATE, IS_FP32_DEST_ACC_EN, SCALE_EN, CLAMP_NEGATIVE, ITERATIONS, DST_IDX, VECTOR_MODE, SCALE) \
    SFPU_CALL(                                                                                               \
        DST_SYNC_MODE,                                                                                       \
        DST_ACCUM_MODE,                                                                                      \
        FN,                                                                                                  \
        (APPROXIMATE, IS_FP32_DEST_ACC_EN, SCALE_EN, ITERATIONS, CLAMP_NEGATIVE),                            \
        DST_IDX,                                                                                             \
        VECTOR_MODE,                                                                                         \
        SCALE)

// ----- runtime-mode invocation aliases (LLK wrapper layer) ------------------

#define SFPU_ONE_PARAM_KERNEL(FN, APPROXIMATE, DST_IDX, VECTOR_MODE) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE), DST_IDX, VECTOR_MODE)

#define SFPU_TWO_PARAM_KERNEL(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, T2), DST_IDX, VECTOR_MODE)

#define SFPU_TWO_PARAM_KERNEL_ONE_RUNTIME(FN, APPROXIMATE, T2, DST_IDX, VECTOR_MODE, PARAM0) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, T2), DST_IDX, VECTOR_MODE, PARAM0)

#define SFPU_THREE_PARAM_KERNEL_ITER_FIRST(FN, APPROXIMATE, ITER, FP32, DST_IDX, VECTOR_MODE) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, ITER, FP32), DST_IDX, VECTOR_MODE)

#define SFPU_THREE_PARAM_KERNEL_FP32_FIRST(FN, APPROXIMATE, FP32, ITER, DST_IDX, VECTOR_MODE) \
    SFPU_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, (APPROXIMATE, FP32, ITER), DST_IDX, VECTOR_MODE)

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
