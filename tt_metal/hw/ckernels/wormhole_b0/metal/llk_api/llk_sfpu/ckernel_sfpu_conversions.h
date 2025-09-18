// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
        result = std::numeric_limits<int32_t>::max();
    }
    v_else {
        // extract mantissa
        sfpi::vInt man = exman8(in);
        // shift the mantissa by (23-exponent) to the right
        sfpi::vInt shift = exp - 23;  // 23 is number of mantissa bits in float32
        man = sfpi::reinterpret<sfpi::vInt>(shft(sfpi::reinterpret<sfpi::vUInt>(man), shift));

        result = man;
    }
    v_endif;
    return result;
}
