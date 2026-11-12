// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

inline void hardshrink_init() {
    // The two-row body advances Dest from its second store through ADDR_MOD_6 (dest.incr = 4: two sfpi
    // rows of two counter steps each), so the body needs no TTINCRWC. Blackhole has no addr_mod base, so
    // the store names ADDR_MOD_6 directly, as square does.
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief hardshrink(x, lambda) = x if |x| > lambda, else 0, over ITERATIONS Dest rows.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @param param0: lambda as fp32 bits.
 * @note Call @ref hardshrink_init first: the body advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_hardshrink(uint32_t param0) {
    static_assert(ITERATIONS % 2 == 0, "calculate_hardshrink() processes dest rows in pairs.");
    // Hardshrink(x, λ) = x if |x| > λ, else 0
    // Single comparison using abs: setsgn(v, 0) clears sign bit
    // param0 contains lambda as FP32 bits. For BF16 inputs, the host pre-rounds
    // lambda to BF16 precision (then re-expands to FP32) so that FP32→FP19b
    // truncation on SFPU preserves the BF16 value exactly. For FP32 inputs,
    // lambda is passed as full FP32, so both input and lambda undergo the same
    // FP32→FP19b truncation, keeping comparisons consistent.
    sfpi::vFloat lambda = Converter::as_float(param0);
    // Two rows per body, 5 SFPU instructions per row: the second row's store carries the Dest
    // increment (ADDR_MOD_6 from hardshrink_init), which drops the per-row TTINCRWC.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat v0 = sfpi::dst_reg[0];
        sfpi::vFloat v1 = sfpi::dst_reg[1];
        sfpi::vFloat abs_v0 = sfpi::setsgn(v0, 0);
        sfpi::vFloat abs_v1 = sfpi::setsgn(v1, 0);
        v_if(abs_v0 <= lambda) { sfpi::dst_reg[0] = 0.0f; }
        v_endif;
        v_if(abs_v1 <= lambda) { sfpi::dst_reg[1].mode(ADDR_MOD_6) = 0.0f; }
        v_endif;
    }
}

}  // namespace ckernel::sfpu
