// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "ckernel_sfpu_converter.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

inline void heaviside_init() {
    // calculate_heaviside's last store steps Dest by one SFPU pass through ADDR_MOD_6.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief heaviside(v) = `value` at v == 0 (either zero), 0.0 for a set sign bit, 1.0 otherwise.
 *
 * The sign is read from the raw bits, so infinities, denormals and NaN dispatch on their sign bit
 * like any other non-zero input.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of SFPU passes (two Dest rows each) to process.
 * @param value: The value returned at v == 0, as fp32 bits.
 * @note Call @ref heaviside_init first: the last store advances Dest through the ADDR_MOD_6 it
 *       programs ({dest.incr = SFP_ROWS}); nothing may rewrite it in between.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_heaviside(const std::uint32_t value) {
    const sfpi::vFloat s = Converter::as_float(value);
    const sfpi::vFloat neg_zero = -0.0f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat v = sfpi::dst_reg[0];

        sfpi::dst_reg[0] = sfpi::vFloat(1.0f);
        v_if(sfpi::as<sfpi::vInt>(v) <= 0) {  // sign bit set, or +0
            sfpi::dst_reg[0] = sfpi::vFloat(0.0f);
            v_and(v >= neg_zero);  // the two zeros
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = s;
        }
        v_endif;
    }
}

}  // namespace sfpu
}  // namespace ckernel
