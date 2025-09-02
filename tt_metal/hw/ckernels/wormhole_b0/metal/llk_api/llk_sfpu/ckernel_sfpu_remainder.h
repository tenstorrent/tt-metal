// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"
#include <type_traits>
#include <cstdint>

using namespace sfpi;
namespace ckernel {
namespace sfpu {

// Type aliases for supported data types
using SupportedDataTypes = std::tuple<float, int32_t>;

/**
 * @brief Initialize remainder operation configuration
 *
 * Loads configuration values into SFPU registers for remainder calculation.
 * The values are stored as 32-bit words but interpreted differently based on DataType:
 * - For float: Interpreted as floating-point values
 * - For int32_t: Reinterpreted as signed 32-bit integers
 *
 * @tparam APPROXIMATION_MODE Whether to use approximation mode (for floating-point operations)
 * @tparam DataType Data type for computation (float or int32_t)
 * @param value First operand value (dividend or input value)
 * @param recip Second operand value (divisor or reciprocal)
 */
template <bool APPROXIMATION_MODE, typename DataType = float>
inline void init_remainder(const uint value, const uint recip) {
    // Compile-time check for supported data types
    static_assert(
        std::is_same_v<DataType, float> || std::is_same_v<DataType, int32_t>,
        "DataType must be either float or int32_t");

    // load vConstFloatPrgm0 = value (interpreted based on DataType)
    _sfpu_load_config32_(0xC, (value >> 16) & 0xFFFF, value & 0xFFFF);
    // load vConstFloatPrgm1 = recip/divisor (interpreted based on DataType)
    _sfpu_load_config32_(0xD, (recip >> 16) & 0xFFFF, recip & 0xFFFF);
}

/**
 * @brief Main remainder calculation function that dispatches to appropriate implementation based on data type
 *
 * @tparam APPROXIMATION_MODE Whether to use approximation mode (for floating-point operations)
 * @tparam ITERATIONS Number of iterations to process
 * @tparam DataType Data type for computation (float or int32_t)
 * @param value Input value for remainder calculation
 * @param recip Reciprocal or divisor value
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8, typename DataType = float>
inline void calculate_remainder(const uint value, const uint recip) {
    // Compile-time check for supported data types
    static_assert(
        std::is_same_v<DataType, float> || std::is_same_v<DataType, int32_t>,
        "DataType must be either float or int32_t");

    if constexpr (std::is_same_v<DataType, float>) {
        // Original floating-point implementation
        calculate_remainder_float<APPROXIMATION_MODE, ITERATIONS>(value, recip);
    } else if constexpr (std::is_same_v<DataType, int32_t>) {
        // New int32 implementation
        calculate_remainder_int32<APPROXIMATION_MODE, ITERATIONS>(value, recip);
    }
}

/**
 * @brief Floating-point remainder calculation implementation (original)
 *
 * @tparam APPROXIMATION_MODE Whether to use approximation mode for floating-point operations
 * @tparam ITERATIONS Number of iterations to process
 * @param value Input value for remainder calculation
 * @param recip Reciprocal value for division
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder_float(const uint value, const uint recip) {
    // SFPU microcode
    vFloat s = vConstFloatPrgm0;
    vFloat recip_val = vConstFloatPrgm1;
    vFloat value_tmp = s;
    s = sfpi::abs(s);
    recip_val = sfpi::abs(recip_val);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat val = dst_reg[0];
        vFloat v = sfpi::abs(val);

        vFloat quotient;
        vInt exp = exexp(v * recip_val);
        v_if(exp < 0) { quotient = vConst0; }
        // Since fp32 has 23 mantissa bits, the LSB represents the fractional part when exp < 23.
        // We effectively round off the fractional bits to zero by right shifting using (exp - 23) and then left
        // shifting it back using (0 - (exp - 23)).
        v_elseif(exp < 23) {
            quotient =
                reinterpret<vFloat>(shft((shft(reinterpret<vUInt>(v * recip_val), (exp - 23))), (0 - (exp - 23))));
        }
        v_else { quotient = v * recip_val; }
        v_endif

        v_if(quotient > v * recip_val) {
            quotient = quotient - 1;
        }
        v_endif;
        v = v - quotient * s;

        v_if(val < 0 && v != 0) { v = s - v; }
        v_endif;

        v_if(value_tmp < 0 && v != 0) { v = v + value_tmp; }
        v_endif;
        v = setsgn(v, value_tmp);
        v_if(s == 0) { v = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;

        constexpr auto iter = 10;
        for (int l = 0; l < iter; l++) {
            v_if(v >= s) { v = s - v; }
            v_endif;
        }
        v_if(sfpi::abs(v) - s == 0.0f) { v = 0.0f; }
        v_endif;
        dst_reg[0] = v;
        dst_reg++;
    }
}

/**
 * @brief Int32 remainder calculation implementation (new)
 *
 * Implements integer remainder operation using the mathematical definition:
 * remainder = dividend - (dividend / divisor) * divisor
 *
 * @tparam APPROXIMATION_MODE Not used for int32 operations, but kept for consistency
 * @tparam ITERATIONS Number of iterations to process
 * @param value Input value for remainder calculation (unused, uses config values)
 * @param recip Divisor value (unused, uses config values)
 */
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_remainder_int32(const uint value, const uint recip) {
    // Convert config values to int32
    // Note: We reinterpret the float config values as int32 values
    vInt divisor = reinterpret<vInt>(vConstFloatPrgm1);

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vInt dividend = reinterpret<vInt>(dst_reg[0]);

        // Handle division by zero case
        vInt result;
        v_if(divisor == 0) {
            // For division by zero, return the original dividend (undefined behavior)
            result = dividend;
        }
        v_else {
            // For int32 remainder: result = dividend % divisor
            // Use the mathematical definition: remainder = dividend - (dividend / divisor) * divisor
            vInt quotient = dividend / divisor;  // Integer division (truncates towards zero)
            result = dividend - quotient * divisor;

            // Note: The above calculation already handles negative numbers correctly
            // for C-style remainder operation where result has same sign as dividend
        }
        v_endif;

        // Store result back (reinterpreted as float for register compatibility)
        dst_reg[0] = reinterpret<vFloat>(result);
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
