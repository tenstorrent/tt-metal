// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

/*
 * Keep macro preconditions outside the tt-llk params wrapper. DST_SYNC is
 * explicit so tests and non-standard kernel preludes can supply their own
 * sync mode. Dest accumulation mode is read at runtime inside
 * get_dest_max_tiles_rt so it stays correct after enable/disable_fp32_dest_acc.
 * Dest-acc-sensitive kernels take is_fp32_dest_acc_en in TEMPLATES, not as a
 * call-macro argument.
 */

namespace ckernel {

template <DstSync DST_SYNC>
inline __attribute__((always_inline)) void _sfpu_check_(
    std::uint32_t dst_index, [[maybe_unused]] VectorMode vector_mode) {
    LLK_ASSERT(
        (dst_index < get_dest_max_tiles_rt<DST_SYNC, DstTileShape::Tile32x32>()), "dst_index exceeds max dest tiles");
}

}  // namespace ckernel

// Strip the parentheses around the template-argument tuple passed to SFPU_UNARY_CALL.

#define _SFPU_EXPAND(...) __VA_ARGS__

/*
 * Macro hygiene: DST_IDX and VECTOR_MODE are evaluated by both the check and
 * params call. Keep call sites to identifiers/literals, not side effects.
 */
#define SFPU_UNARY_CALL(DST_SYNC, FN, TEMPLATES, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC>(DST_IDX, VECTOR_MODE),               \
     _llk_math_eltwise_unary_sfpu_params_(                                  \
         ::ckernel::sfpu::FN<_SFPU_EXPAND TEMPLATES>, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

// Non-templated functor in `ckernel::sfpu`.
#define SFPU_UNARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC, FN, DST_IDX, VECTOR_MODE, ...) \
    (::ckernel::_sfpu_check_<DST_SYNC>(DST_IDX, VECTOR_MODE),                     \
     _llk_math_eltwise_unary_sfpu_params_(::ckernel::sfpu::FN, DST_IDX, VECTOR_MODE, ##__VA_ARGS__))

/*
 * SFPU init macros (3 total)
 *
 * No dst index involved, so no bound check and no DST_SYNC argument.
 * The bare init takes the dest accumulation mode explicitly: pass DST_ACCUM_MODE
 * for the kernel compile-time default, or the op's own is_fp32_dest_acc_en when the
 * active dest mode may differ from it (e.g. after set_fp32_dest_acc).
 * Argument order for callback inits: OP first (which selects the SfpuType),
 * then the init callback (the FN-like argument), then the templates that
 * parameterise it.
 */

/*
 * Bare init: no callback.
 *   SFPU_UNARY_INIT(abs, DST_ACCUM_MODE);
 *   SFPU_UNARY_INIT(softcap, is_fp32_dest_acc_en);
 */
#define SFPU_UNARY_INIT(OP, ACCUM) ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP, ACCUM>()

/*
 * Init with a templated callback.
 *   SFPU_UNARY_INIT_FN(erf, sfpu::erf_init, (APPROXIMATE));
 *   SFPU_UNARY_INIT_FN(log, sfpu::log_init, (APPROX, fp32, FAST));
 */
#define SFPU_UNARY_INIT_FN(OP, INIT_FN, TEMPLATES) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>)

/*
 * Init with a templated callback and extra runtime arguments.
 *   SFPU_UNARY_INIT_FN_ARGS(exponential, sfpu::exp_init, (APPROX), scale, clamp_neg);
 */
#define SFPU_UNARY_INIT_FN_ARGS(OP, INIT_FN, TEMPLATES, ...) \
    ::ckernel::llk_math_eltwise_unary_sfpu_init<::SfpuType::OP>(INIT_FN<_SFPU_EXPAND TEMPLATES>, ##__VA_ARGS__)
