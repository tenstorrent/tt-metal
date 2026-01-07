// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_rsqrt.h"
#include "sfpi_fp16.h"

namespace ckernel::sfpu {

// Calculate: result = rsqrt(x + param0)
// param0 is the bit representation of a float
// This is useful for operations like RMSNorm: rsqrt(variance + epsilon)
template <bool APPROXIMATION_MODE, int ITERATIONS, bool fp32_dest_acc_en, bool FAST_APPROX>
inline void calculate_add_rsqrt(uint32_t param0) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_plus_addend = x + Converter::as_float(param0);

        // Use the rsqrt body function (RECIPROCAL=true for rsqrt)
        sfpi::vFloat y = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(x_plus_addend);

        if constexpr (fp32_dest_acc_en) {
            sfpi::dst_reg[0] = y;
        } else {
            sfpi::dst_reg[0] = sfpi::reinterpret<sfpi::vFloat>(float_to_fp16b(y, 0));
        }
        sfpi::dst_reg++;
    }
}

// Initialize for add + rsqrt operation (just initializes rsqrt constants)
template <bool APPROXIMATION_MODE>
inline void init_add_rsqrt() {
    _init_sqrt_<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
