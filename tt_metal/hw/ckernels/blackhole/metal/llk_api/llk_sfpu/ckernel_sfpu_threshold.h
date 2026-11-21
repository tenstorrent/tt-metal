// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

/**
 * @brief Sets up the threshold kernel's Dest walk and resets the Dest counters.
 *
 * Programs ADDR_MOD_6 (dest.incr = 4): the two-row body advances Dest from its second store instead of a
 * separate increment, so a missing init misaddresses Dest without an error.
 *
 * @note Call before @ref _calculate_threshold_; nothing may rewrite ADDR_MOD_6 in between.
 */
inline void threshold_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 4}}.set(ADDR_MOD_6);
    math::reset_counters(p_setrwc::SET_ABD_F);
}

/**
 * @brief Replaces every Dest value <= threshold with value, over ITERATIONS rows from the current Dest row.
 *
 * @tparam APPROXIMATION_MODE: Unused; the kernel is exact in both modes.
 * @tparam ITERATIONS: Number of Dest rows to process; must be even.
 * @tparam T: float, or std::uint32_t holding fp32 bits.
 * @param threshold: Values at or below it are replaced.
 * @param value: The replacement.
 * @note Call @ref threshold_init first: the body advances Dest through ADDR_MOD_6.
 */
template <bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _calculate_threshold_(T threshold, T value) {
    static_assert(is_supported_threshold_type_v<T>, "Type T must be either float or uint32_t");
    static_assert(ITERATIONS % 2 == 0, "_calculate_threshold_ processes dest rows in pairs.");

    sfpi::vFloat v_threshold;
    sfpi::vFloat v_value;
    if constexpr (std::is_same_v<T, float>) {
        v_threshold = threshold;
        v_value = value;
    } else if constexpr (std::is_same_v<T, std::uint32_t>) {
        v_threshold = Converter::as_float(threshold);
        v_value = Converter::as_float(value);
    }
    // Two rows per body, 4 SFPU instructions per row: the second row's store carries the Dest
    // increment (ADDR_MOD_6 from threshold_init), which drops the per-row TTINCRWC.
#pragma GCC unroll 4
    for (int d = 0; d < ITERATIONS; d += 2) {
        sfpi::vFloat in0 = sfpi::dst_reg[0];
        sfpi::vFloat in1 = sfpi::dst_reg[1];
        v_if(in0 <= v_threshold) { sfpi::dst_reg[0] = v_value; }
        v_endif;
        v_if(in1 <= v_threshold) { sfpi::dst_reg[1].mode(ADDR_MOD_6) = v_value; }
        v_endif;
    }
}

}  // namespace ckernel::sfpu
