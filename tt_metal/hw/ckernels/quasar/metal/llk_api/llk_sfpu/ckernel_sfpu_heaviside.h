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
    // The row's last store steps Dest by one SFPU pass (SFP_ROWS = 2 counter rows, the step sfpi's
    // dst_reg++ takes), so the body needs no TTINCRWC. Same ADDR_MOD_6 value as init_typecast. The
    // SFPU load/store addr_mod field is 3 bits wide on Quasar, so the body names slot 6 directly.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief heaviside(v) = `value` at v == 0 (either zero), 0.0 for a set sign bit, 1.0 otherwise.
 *
 * 7 SFPU instructions per row, written as three stores that narrow the enabled lanes: 1.0 to every
 * lane; 0.0 to the lanes whose bits, read as a two's-complement integer, are <= 0 (a set sign bit, or
 * +0); then `value` to those of them that compare >= -0.0 as floats, which leaves the two zeros
 * (v_and only narrows the lanes the v_if enabled). The previous form (copysgn + add, then an
 * sfpi::abs(v) == 0 arm) cost 11: an SFPLOADI, SFPMOV + SFPSETSGN + SFPADDI for the sign arm,
 * SFPABS + SFPSETCC for the zero test, an SFPMOV merge, a TTINCRWC and a rolled RISC loop.
 *
 * The sign comes from the integer compare, not a float one, so NaN lands where it did before: a NaN
 * with a clear sign bit is 1.0, with a set one 0.0. A float compare cannot do that; it is false for a
 * NaN of either sign. The -0.0 bound keeps -0.0 in the `value` lanes whether the float compare orders
 * -0.0 below +0.0 or treats them as equal.
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
        v_if(sfpi::as<sfpi::vInt>(v) <= 0) {
            sfpi::dst_reg[0] = sfpi::vFloat(0.0f);
            v_and(v >= neg_zero);
            sfpi::dst_reg[0].mode(ADDR_MOD_6) = s;
        }
        v_endif;
    }
}

}  // namespace sfpu
}  // namespace ckernel
