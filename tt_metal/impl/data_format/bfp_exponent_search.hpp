// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <span>

namespace tt::tt_metal::detail {

// Compare the squared errors from Emax and Emax-1. Keep Emax if the errors are equal.
// Use double precision to prevent overflow when the input contains large FP32 values.
template <unsigned MagnitudeBits, typename Round>
bool prefer_lower_bfp_exponent(std::span<const uint32_t> row, uint8_t max_exp, Round&& round) {
    static_assert(MagnitudeBits == 3 || MagnitudeBits == 7);
    // Keep the usual exponent for zero, very small values, infinity, and NaN.
    // A smaller exponent could cause underflow when the device reads the values.
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
        // The rounding function requires the input exponent to be no larger than the shared exponent.
        // Limit larger magnitudes to the maximum value that the smaller exponent can represent.
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
