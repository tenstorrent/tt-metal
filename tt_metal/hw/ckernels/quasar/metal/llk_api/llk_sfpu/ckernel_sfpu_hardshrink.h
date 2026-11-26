// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

/**
 * @brief Programs ADDR_MOD_6 = {dest.incr = 2 * SFP_ROWS}, two sfpi rows per body of @ref calculate_hardshrink.
 *
 * @note ADDR_MOD_6 is shared: call this after any other init that programs it.
 */
inline void hardshrink_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

// The param0 the generated hardshrink kernel is fitted for, as FP32 bits; any other param0 runs the stock kernel.
inline constexpr std::uint32_t hardshrink_bf16_param0_bits = 0x3f000000u;
// Whether BF16 DEST runs the generated hardshrink kernel as one call over the whole tile.
inline constexpr bool hardshrink_bf16_whole_tile = false;

/**
 * @brief hardshrink(x, lambda) = x if |x| > lambda, else 0, over ITERATIONS Dest rows (must be even).
 *
 * @note Call @ref hardshrink_init first: the second store of each row pair advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en = true>
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
