// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
// The leaf header reaches for the exp helpers unqualified ("ckernel_sfpu_exp.h"); they live
// in llk_api/llk_sfpu, one layer up from tt-llk. Pulled in here under their on-path spelling
// so the dependency is visible at the point that needs it.
#include "ckernel_sfpu_exp.h"
#include "sfpu/experimental/ckernel_sfpu_sdpa_exp_unclamped.h"

namespace ckernel::sfpu {

/**
 * @brief Exponentiate one DEST face in place, without the upper input clamp.
 *
 * Wraps @ref _ckernel_sfpu_exp_accurate_upper_unclamped_ in the dst_reg walk the SFPU dispatch
 * expects, so the kernel can be driven through @ref _llk_math_eltwise_unary_sfpu_params_ /
 * SFPU_UNARY_CALL like any other unary op. VectorMode::RC repeats it over the four faces.
 *
 * @tparam SCALE_EN: Multiply the input by exp_base_scale_factor first, values = <true/false>
 * @param exp_base_scale_factor: Scale as a raw bf16 bit pattern; ignored when SCALE_EN is false.
 * @note bf16 DEST only -- the leaf static_asserts on DST_ACCUM_MODE.
 * @note Callers must pass val <= 0, which is what makes dropping the upper clamp safe. The
 *       clamped path saturates xlog2 = val/ln2 + 127 at its upper bound; that bound is
 *       unreachable for non-positive inputs, so removing it is dead-code removal for the SDPA
 *       use case and a wrap in _float_to_int32_for_exp_21f_ for anything above it.
 * @note No op-specific init: this is a pure sfpi leaf that materialises every constant as an
 *       SFPLOADI immediate, so the invariant SFPU config + ADDR_MOD_7 is all it needs. Calling
 *       exp_init would only program the TTI exp path's state, which nothing here reads.
 */
template <bool SCALE_EN>
inline void calculate_sdpa_exp_unclamped(const std::uint32_t exp_base_scale_factor) {
    // One SFPU slot is 4 DEST rows x 8 columns, so a full 16x16 face is 8 slots.
    constexpr int ITERATIONS_FULL_FACE = 8;
    for (int d = 0; d < ITERATIONS_FULL_FACE; d++) {
        const sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] =
            _ckernel_sfpu_exp_accurate_upper_unclamped_<SCALE_EN, DST_ACCUM_MODE>(val, exp_base_scale_factor);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
