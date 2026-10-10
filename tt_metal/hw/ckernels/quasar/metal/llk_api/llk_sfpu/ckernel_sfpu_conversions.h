// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

#include "sfpi.h"
#include "sfpu/ckernel_sfpu_operand.h"

namespace ckernel {
namespace sfpu {

// Helper function for _sfpu_binary_power_
// This function is based on _float32_to_int32_, but expects a positive input, which simplifies the code
// and makes it faster
sfpi_inline sfpi::vInt _float_to_int32_positive_(sfpi::vFloat in) {
    sfpi::vInt result;
    sfpi::vInt exp = exexp(in);  // extract exponent
    v_if(exp < 0) { result = 0; }
    v_elseif(exp > 30)  // overflow occurs above this range
    {
        // set to int32 max value in case of overflow
        result = std::numeric_limits<std::int32_t>::max();
    }
    v_else {
        // extract mantissa
        sfpi::vInt man = exman(in, sfpi::MantissaMode::ImplicitOne);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23;  // 23 is number of mantissa bits in float32
        man = shft(man, shift, sfpi::ShiftMode::Logical);

        result = man;
    }
    v_endif;
    return result;
}

/**
 * @brief Converts float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE).
 * Implements the "add 0x7fff + LSB" algorithm for correct tie-breaking, ported
 * from BH. Applied in software before SFPSTORE because SFPSTORE truncates
 * fp32->bf16 on all architectures.
 *
 * @param in: float32 value to convert
 * @return bf16 value packed in the upper 16 bits of a float32
 */
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::as<sfpi::vUInt>(in);

    // Extract the LSB of what will become the bf16 mantissa (bit 16 of float32).
    // Needed for the tie-breaker: round to even.
    sfpi::vUInt lsb = (bits >> 16) & 1;

    // Add 0x7fff + lsb to implement RNE:
    // - lower 16 bits > 0x8000      -> overflow, rounds up
    // - lower 16 bits < 0x8000      -> no overflow, rounds down
    // - lower 16 bits == 0x8000 (tie)
    //     and lsb=0: 0x7fff+0=0xffff, no overflow -> stays even
    //     and lsb=1: 0x7fff+1=0x8000,    overflow -> rounds up to even
    bits = bits + 0x7fffU + lsb;

    // Clear the lower 16 bits to get bf16 in upper 16 bits (bf16 format in float32).
    bits = bits & 0xFFFF0000U;

    return sfpi::as<sfpi::vFloat>(bits);
}

/**
 * @brief Dest store policy that applies @ref float32_to_bf16_rne before SFPSTORE when ROUND_TO_BF16.
 */
template <bool ROUND_TO_BF16>
struct DestBf16RneFormat : SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat> {
    template <SfpuReg REG>
    sfpi_inline static void store(int index, sfpi::vFloat value) {
        static_assert(REG == SfpuReg::Dest, "bf16 RNE store policy is Dest-only");
        if constexpr (ROUND_TO_BF16) {
            value = float32_to_bf16_rne(value);
        }
        SfpiFormat<sfpi::DataLayout::Default, sfpi::vFloat>::template store<REG>(index, value);
    }
};

}  // namespace sfpu
}  // namespace ckernel
