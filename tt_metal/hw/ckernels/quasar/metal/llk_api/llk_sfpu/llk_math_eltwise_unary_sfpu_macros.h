// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu.h"

// Quasar keeps the same macro surface as BH/WH. DST_SYNC and DST_ACCUM are
// unused until Quasar has an equivalent of get_dest_max_tiles<...>.

namespace ckernel {

template <DstSync /*DST_SYNC*/, bool /*DST_ACCUM*/>
inline __attribute__((always_inline)) void _sfpu_check_(
    [[maybe_unused]] std::uint32_t dst_index, VectorMode vector_mode) {
    LLK_ASSERT(
        vector_mode == VectorMode::R || vector_mode == VectorMode::C || vector_mode == VectorMode::RC ||
            vector_mode == VectorMode::None || vector_mode == VectorMode::RC_custom,
        "Quasar SFPU supports vector modes R, C, RC, None, RC_custom");
}

}  // namespace ckernel

// Strip the parentheses around the template-argument tuple passed to SFPU_UNARY_CALL.

#define _SFPU_EXPAND(...) __VA_ARGS__

// Macro hygiene: DST_IDX and VECTOR_MODE are evaluated by both the check and
// params call. Keep call sites to identifiers/literals, not side effects.
#define SFPU_UNARY_CALL(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC, DST_ACCUM>(DST_IDX, VECTOR_MODE),               \
     _llk_math_eltwise_unary_sfpu_params_(                                             \
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

// Non-templated functor in `ckernel::sfpu`.
#define SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC, DST_ACCUM, FN, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC, DST_ACCUM>(DST_IDX, VECTOR_MODE),                     \
     _llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

// Init macros take OP first, then the optional init callback and template args.

// Init with an optional non-templated init function.
//   SFPU_UNARY_INIT(abs);                                       // no init function
//   SFPU_UNARY_INIT(greater_than_zero, sfpu::init_zero_comp);  // non-templated init function
#define SFPU_UNARY_INIT_1(OP) ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>()
#define SFPU_UNARY_INIT_2(OP, INIT_FN) ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN)
#define SFPU_UNARY_INIT_PICK(_1, _2, NAME, ...) NAME
#define SFPU_UNARY_INIT(...) \
    SFPU_UNARY_INIT_PICK(    \
        __VA_ARGS__, SFPU_UNARY_INIT_2, SFPU_UNARY_INIT_1, _ignore /* at least one argument */)(__VA_ARGS__)

// Init with a templated callback.
//   SFPU_UNARY_INIT_FN(erf, sfpu::erf_init, (APPROXIMATE));
//   SFPU_UNARY_INIT_FN(log, sfpu::log_init, (APPROX, fp32, FAST));
#define SFPU_UNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>)

// Init with a templated callback and extra runtime arguments.
//   SFPU_UNARY_INIT_FN_ARGS(exponential, sfpu::exp_init, (APPROX), scale, clamp_neg);
#define SFPU_UNARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>, ##__VA_ARGS__)
