// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu.h"

// Quasar keeps the same macro surface as BH/WH.

namespace ckernel {

template <DstSync DST_SYNC, bool DST_ACCUM, trisc::DstTileShape TILE_SHAPE = trisc::DstTileShape::Tile32x32>
inline __attribute__((always_inline)) void _sfpu_check_(std::uint32_t dst_index, VectorMode vector_mode) {
    LLK_ASSERT(
        (dst_index < trisc::get_dest_max_tiles<DST_SYNC, DST_ACCUM, TILE_SHAPE>()), "dst_index exceeds max dest tiles");
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
// Shared signature matches WH/BH (no DST_ACCUM slot). Inject DST_ACCUM_MODE
// for dest-tile bounds; Quasar has no runtime dest-acc switch.
#define SFPU_UNARY_CALL_QSR(DST_SYNC, DST_ACCUM, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC, DST_ACCUM>(DST_IDX, VECTOR_MODE),                   \
     _llk_math_eltwise_unary_sfpu_params_(                                                 \
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

#define SFPU_UNARY_CALL(DST_SYNC, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    SFPU_UNARY_CALL_QSR(DST_SYNC, DST_ACCUM_MODE, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ##__VA_ARGS__)

// Templated functor in `ckernel::sfpu` operating on a non-default Dest tile shape.
#define SFPU_UNARY_CALL_TINY_TILE_QSR(DST_SYNC, DST_ACCUM, TILE_SHAPE, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC, DST_ACCUM, TILE_SHAPE>(DST_IDX, VECTOR_MODE),                             \
     _llk_math_eltwise_unary_sfpu_params_<TILE_SHAPE>(                                                           \
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

#define SFPU_UNARY_CALL_TINY_TILE(DST_SYNC, TILE_SHAPE, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    SFPU_UNARY_CALL_TINY_TILE_QSR(                                                                \
        DST_SYNC, DST_ACCUM_MODE, TILE_SHAPE, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ##__VA_ARGS__)

// Non-templated functor in `ckernel::sfpu`.
#define SFPU_UNARY_CALL_NO_TEMPLATE_ARGS_QSR(DST_SYNC, DST_ACCUM, FN, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC, DST_ACCUM>(DST_IDX, VECTOR_MODE),                         \
     _llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

#define SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC, FN, DST_IDX, VECTOR_MODE, ...) \
    SFPU_UNARY_CALL_NO_TEMPLATE_ARGS_QSR(DST_SYNC, DST_ACCUM_MODE, FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__)

// Init macros take OP first, then the accum mode / init callback and template args.

// Bare init: no callback. ACCUM is accepted for signature parity with WH/BH and
// deliberately unused -- Quasar's llk_math_eltwise_unary_sfpu_init takes no
// dest-acc parameter, and Quasar has no runtime dest-acc switch.
//   SFPU_UNARY_INIT(abs, DST_ACCUM_MODE);
#define SFPU_UNARY_INIT(OP, ACCUM) ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>()

// Init with a non-templated callback (mirrors SFPU_BINARY_INIT_FN_NO_ARGS /
// SFPU_TERNARY_INIT_FN_NO_ARGS).
//   SFPU_UNARY_INIT_FN_NO_ARGS(greater_than_zero, sfpu::init_zero_comp);
#define SFPU_UNARY_INIT_FN_NO_ARGS(OP, INIT_FN) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN)

// Init with a templated callback.
//   SFPU_UNARY_INIT_FN(erf, sfpu::erf_init, (APPROXIMATE));
//   SFPU_UNARY_INIT_FN(log, sfpu::log_init, (APPROX, fp32, FAST));
#define SFPU_UNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>)

// Init with a templated callback and extra runtime arguments.
//   SFPU_UNARY_INIT_FN_ARGS(exponential, sfpu::exp_init, (APPROX), scale, clamp_neg);
#define SFPU_UNARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::ckernel::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>, ##__VA_ARGS__)
