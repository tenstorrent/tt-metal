// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

namespace tt::tt_metal::detail {

// Score the existing B-format rounder in double precision. Even the square of
// the largest finite FP32 value fits in double. Keep Emax on exact ties.
template <unsigned MagnitudeBits, typename Round>
bool prefer_lower_bfp_exponent(std::span<const uint32_t> row, uint8_t max_exp, Round&& round) {
    static_assert(MagnitudeBits == 3 || MagnitudeBits == 7);
    // Preserve legacy zero/denormal/near-underflow and Inf/NaN behavior. The
    // smaller exponent must not underflow when the packed value is unpacked.
    if (max_exp <= MagnitudeBits || max_exp == 255) {
        return false;
    }
    constexpr uint32_t magnitude_mask = (1u << MagnitudeBits) - 1;
    const uint8_t lower_exp = max_exp - 1;
    const double scale = std::ldexp(1.0, int(max_exp) - 127 - (int(MagnitudeBits) - 1));
    double ordinary_error = 0.0;
    double lower_error = 0.0;
    for (uint32_t value : row) {
        const uint32_t exponent = (value >> 23) & 0xff;
        const uint32_t ordinary = round(value, max_exp);
        // The legacy rounder assumes input exp <= shared exp. Saturate larger
        // values explicitly for the lower candidate instead of violating that.
        const uint32_t lower =
            exponent > lower_exp ? ((value >> 31) << MagnitudeBits) | magnitude_mask : round(value, lower_exp);
        const double input = std::bit_cast<float>(value);
        const double q0 = (ordinary & magnitude_mask) * scale * ((ordinary >> MagnitudeBits) ? -1.0 : 1.0);
        const double q1 = (lower & magnitude_mask) * (scale * 0.5) * ((lower >> MagnitudeBits) ? -1.0 : 1.0);
        ordinary_error += (input - q0) * (input - q0);
        lower_error += (input - q1) * (input - q1);
    }
    return lower_error < ordinary_error;
}

}  // namespace tt::tt_metal::detail
