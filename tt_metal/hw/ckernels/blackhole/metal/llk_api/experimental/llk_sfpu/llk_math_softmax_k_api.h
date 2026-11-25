// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/experimental/ckernel_sfpu_softmax_k.h"
#include "sanitizer/api.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Initialize the SFPU for @ref llk_math_sfpu_softmax_k (programs the exponential's constants).
 *
 * @tparam is_fp32_dest_acc_en: DEST width the matching llk_math_sfpu_softmax_k call will run with.
 */
template <bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_softmax_k_init() {
    SAN_HOOK(unsupported());
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused>(_init_softmax_k_<is_fp32_dest_acc_en>);
}

/**
 * @brief Per-row softmax over the first k columns of DEST rows 0-3 of tile 0, using the row maxima in rows 8-11.
 *
 * Columns >= k are padding: their input may hold anything (inf / NaN included) and is excluded from the row sum, and
 * their output is 0.0. See @ref _softmax_k_.
 *
 * @tparam k: number of valid columns per row, 1..16.
 * @tparam is_fp32_dest_acc_en: DEST width.
 * @note Run @ref llk_math_sfpu_softmax_k_init first.
 */
template <int k, bool is_fp32_dest_acc_en>
inline void llk_math_sfpu_softmax_k() {
    SAN_HOOK(unsupported());
    _llk_math_eltwise_unary_sfpu_params_(_softmax_k_<k, is_fp32_dest_acc_en>, 0, VectorMode::RC_custom);
}

}  // namespace sfpu
}  // namespace ckernel
