// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_macros.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_log.h"
#endif

/**
 * Fused Gumbel scoring: one SFPI pass computing, per DST datum,
 *
 *     score = logits * inv_temperature + (-log(-log(U)))
 *
 * DST layout contract: the uniform noise tile U sits at `idst`, the logits
 * tile LOGITS_DST_OFFSET slots above it, and the score overwrites the noise
 * tile in place. U must be strictly inside (0, 1) so both logs stay finite;
 * the caller's rand from/scale bits enforce that.
 *
 * Both logs run through tt-llk's _calculate_log_body_no_init_, which uses
 * immediate constants only -- no programmable const LREGs and no replay
 * slots -- so this pass composes with rand_tile on both architectures
 * (Wormhole's rand programs LREG12/LREG13; nothing here reads them).
 */

namespace ckernel {

#ifdef TRISC_MATH

namespace sfpu {

template <std::uint32_t LOGITS_DST_OFFSET>
inline void _calculate_gumbel_score_(const std::uint32_t inv_temperature_bits) {
    // One DST tile spans 64 rows / SFP_DESTREG_STRIDE = 32 sfpi rows -- same
    // convention as tt-llk's binary SFPU ops.
    constexpr std::uint32_t dst_tile_size_sfpi = 32U;
    const sfpi::vFloat inv_temperature = Converter::as_float(inv_temperature_bits);
#pragma GCC unroll 8
    for (int d = 0; d < 8; d++) {
        const sfpi::vFloat u = sfpi::dst_reg[0];
        const sfpi::vFloat logits = sfpi::dst_reg[LOGITS_DST_OFFSET * dst_tile_size_sfpi];
        // Negations are LREG sign flips; neither intermediate touches DST.
        const sfpi::vFloat neg_log_u = -_calculate_log_body_no_init_(u);
        const sfpi::vFloat gumbel = -_calculate_log_body_no_init_(neg_log_u);
        sfpi::vFloat score = logits * inv_temperature + gumbel;
        if constexpr (!DST_ACCUM_MODE) {
            score = sfpi::convert<sfpi::vFloat16b>(score, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = score;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu

#endif  // TRISC_MATH

/**
 * @brief Initializes the fused Gumbel scoring SFPU operation.
 */
ALWI void gumbel_score_tile_init() { MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused>())); }

/**
 * @brief Overwrites the uniform tile at `idst` with the fused Gumbel score.
 *
 * The logits tile must already sit `logits_dst_offset` DST slots above
 * `idst`. `inv_temperature_bits` is the FP32 bit pattern of 1/temperature.
 */
template <uint32_t logits_dst_offset = 1U>
ALWI void gumbel_score_tile(uint32_t idst, uint32_t inv_temperature_bits) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        _calculate_gumbel_score_,
        (logits_dst_offset),
        idst,
        VectorMode::RC,
        inv_temperature_bits));
}

}  // namespace ckernel
