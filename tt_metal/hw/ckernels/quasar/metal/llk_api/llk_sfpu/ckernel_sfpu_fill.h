// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <type_traits>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "llk_math_eltwise_sfpu_common.h"  // _sfpu_sfpmem_type_<FMT>()
#include "sfpi_constants.h"

namespace ckernel {
namespace sfpu {

constexpr bool _is_int_format_(DataFormat fmt) {
    return fmt == DataFormat::Int32 || fmt == DataFormat::Int16 || fmt == DataFormat::Int8 || fmt == DataFormat::UInt8;
}

// Broadcast a 32-bit pattern to all elements of Dest using a fixed SFPMEM store mode.
template <std::uint32_t SFPMEM_MODE, int ITERATIONS>
inline void _fill_store_(const std::uint32_t bits) {
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_LOWER, bits & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG1, sfpi::SFPLOADI_MOD0_UPPER, bits >> 16);
#pragma GCC unroll 8
    for (std::uint32_t d = 0; d < ITERATIONS; d++) {
        TTI_SFPSTORE(p_sfpu::LREG1, SFPMEM_MODE, ADDR_MOD_7, 0, 0);
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

/**
 * @brief Broadcast a constant to all elements of Dest, via the int or float SFPU store path.
 *
 * @tparam FMT: Target SFPU DataFormat; selects the int store path (Int32/Int16/Int8/UInt8) or
 *         the float bit-cast path otherwise.
 * @tparam IS_32b_DEST_EN: Whether Dest is configured 32-bit (vs 16-bit).
 * @tparam ITERATIONS: Number of SFPU loop iterations.
 * @param value: The constant to broadcast. Its type tracks FMT: `uint32_t` (the raw integer)
 *        for an integer FMT, `float` otherwise — pass the value in FMT's own domain
 */
template <DataFormat FMT, bool IS_32b_DEST_EN, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_fill(const std::conditional_t<_is_int_format_(FMT), std::uint32_t, float> value) {
    if constexpr (_is_int_format_(FMT)) {
        constexpr std::uint32_t SFPMEM_MODE = IS_32b_DEST_EN ? p_sfpu::sfpmem::INT32 : _sfpu_sfpmem_type_<FMT>();
        _fill_store_<SFPMEM_MODE, ITERATIONS>(value);
    } else {
        _fill_store_<p_sfpu::sfpmem::DEFAULT, ITERATIONS>(__builtin_bit_cast(std::uint32_t, value));
    }
}

}  // namespace sfpu
}  // namespace ckernel
