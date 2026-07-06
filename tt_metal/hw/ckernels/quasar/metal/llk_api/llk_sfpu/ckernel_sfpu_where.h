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
 *         @c SFPU_TERNARY_CALL.
 *
 * @param in0_offset DEST offset holding the condition operand.
 * @param in1_offset DEST offset holding the true-branch operand.
 * @param in2_offset DEST offset holding the false-branch operand.
 * @param out_offset DEST offset that receives the per-lane result.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_where(
    const std::uint32_t in0_offset,
    const std::uint32_t in1_offset,
    const std::uint32_t in2_offset,
    const std::uint32_t out_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat cond = sfpi::dst_reg[in0_offset >> 1];
        sfpi::vFloat true_val = sfpi::dst_reg[in1_offset >> 1];
        sfpi::vFloat result = sfpi::dst_reg[in2_offset >> 1];  // load false_val into result reg

        v_if(cond != 0) { result = true_val; }
        v_endif;

        sfpi::dst_reg[out_offset >> 1] = result;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
