// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"
#include "ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

template <typename T>
constexpr bool is_supported_threshold_type_v = std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>;

/**
 * @brief Programs ADDR_MOD_6 = {dest.incr = 2 * SFP_ROWS}, two sfpi rows per body of @ref _calculate_threshold_.
 *
 * @note ADDR_MOD_6 is shared: call this after any other init that programs it.
 */
inline void threshold_init() {
    addr_mod_t{.srca = {.incr = 0}, .srcb = {.incr = 0}, .dest = {.incr = 2 * ckernel::math::SFP_ROWS}}.set(ADDR_MOD_6);
    math::_reset_counters_<p_setrwc::SET_ABD_F>();
}

/**
 * @brief Replaces every Dest value <= threshold with value, over ITERATIONS Dest rows (must be even).
 *
 * @tparam T: float, or std::uint32_t holding fp32 bits.
 * @note Call @ref threshold_init first: the second store of each row pair advances Dest through ADDR_MOD_6.
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
