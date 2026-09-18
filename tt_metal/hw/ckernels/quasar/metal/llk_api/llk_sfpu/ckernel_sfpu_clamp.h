// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/**
 * @brief Element-wise clamp: out = min(max(x, min_val), max_val).
 *
 * Bounds are enforced with v_if/v_elseif compares; min_val/max_val are reinterpreted from their raw
 * 32-bit word to fp32.
 *
 * @tparam APPROXIMATION_MODE: accepted for ABI parity but ignored (clamp is exact).
 * @tparam ITERATIONS: number of SFPU passes over the tile.
 * @param min_val: lower bound as an fp32 bit pattern.
 * @param max_val: upper bound as an fp32 bit pattern.
 * @note The compares lower to SFPMAD against vConstNeg1 (LREG11), so LREG11 must hold -1.0;
 *       @ref _init_sfpu_config_reg_ re-establishes it per SFPU launch.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_clamp(std::uint32_t min_val, std::uint32_t max_val) {
    sfpi::vFloat min_bound = sfpi::as<sfpi::vFloat>(sfpi::vUInt(min_val));
    sfpi::vFloat max_bound = sfpi::as<sfpi::vFloat>(sfpi::vUInt(max_val));
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];

        v_if (val < min_bound) {
            val = min_bound;
        }
        v_elseif (val >= max_bound) {
            val = max_bound;
        }
        v_endif;

        sfpi::dst_reg[0] = val;
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
