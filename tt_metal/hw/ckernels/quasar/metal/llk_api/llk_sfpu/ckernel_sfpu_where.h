// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Per-lane ternary select: @c out = (cond == 0) ? false_val : true_val.
 *
 * Loads @c false_val directly into the @c result variable so sfpi aliases
 * them to the same LREG. The @c cond==0 branch then becomes implicit — only
 * the @c cond!=0 lanes need a CC-gated move to overwrite the result reg
 * with @c true_val. sfpi emits the same sequence as the explicit
 * @c binary_comp pattern:
 *
 *     SFPLOAD cond
 *     SFPLOAD true_val
 *     SFPLOAD false_val (= result reg)
 *     SFPSETCC (cond == 0)
 *     SFPCOMPC
 *     SFPMOV result <- true_val
 *     SFPENCC
 *     SFPSTORE result
 *     TTINCRWC (from dst_reg++) — advances dest counter by one SFP row pair.
 *
 * @tparam APPROXIMATION_MODE Unused for @c where; kept for API parity with
 *         other SFPU kernels.
 * @tparam ITERATIONS         Inner SFPU row-pair count per face. Defaults
 *         to 8 for the standard 16-row face. The outer per-face loop and
 *         section base setup are owned by
 *         @c _llk_math_eltwise_ternary_sfpu_params_.
 *
 * @param dst_index_in0 DEST tile index holding the condition operand.
 * @param dst_index_in1 DEST tile index holding the true-branch operand.
 * @param dst_index_in2 DEST tile index holding the false-branch operand.
 * @param dst_index_out DEST tile index that receives the per-lane result.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_where(
    const std::uint32_t dst_index_in0,
    const std::uint32_t dst_index_in1,
    const std::uint32_t dst_index_in2,
    const std::uint32_t dst_index_out) {
    constexpr std::uint32_t dst_tile_size_sfpi = 32;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat cond = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi];
        sfpi::vFloat true_val = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi];
        sfpi::vFloat result = sfpi::dst_reg[dst_index_in2 * dst_tile_size_sfpi];  // load false_val into result reg

        v_if(cond != 0) { result = true_val; }
        v_endif;

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
