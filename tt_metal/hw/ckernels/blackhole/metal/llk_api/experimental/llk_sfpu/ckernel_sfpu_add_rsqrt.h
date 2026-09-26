// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

// Calculate: result = rsqrt(x * INPUT_SCALE + param0)
// param0 and INPUT_SCALE are the bit representations of floats
// This is useful for operations like RMSNorm: rsqrt(variance + epsilon)
// typed_bf16_store preserves the explicit BF16 destination-store mode used by
// fused normalization callers. Assigning the converted value back to vFloat
// instead retains the default source-format store selected by the SrcB format.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    bool fp32_dest_acc_en,
    bool FAST_APPROX,
    bool typed_bf16_store = false,
    uint32_t INPUT_SCALE = 0x3f800000u>
inline void calculate_add_rsqrt(uint32_t param0) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_plus_addend;
        if constexpr (INPUT_SCALE == 0x3f800000u) {
            x_plus_addend = x + Converter::as_float(param0);
        } else {
            x_plus_addend = x * Converter::as_float(INPUT_SCALE) + Converter::as_float(param0);
        }

        // Use the rsqrt body function (RECIPROCAL=true for rsqrt)
        sfpi::vFloat y = _calculate_sqrt_body_<APPROXIMATION_MODE, true, FAST_APPROX>(x_plus_addend);

        if constexpr (!fp32_dest_acc_en && typed_bf16_store) {
            sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y, RoundMode::Nearest);
        } else {
            if constexpr (!fp32_dest_acc_en) {
                y = sfpi::convert<sfpi::vFloat16b>(y, RoundMode::Nearest);
            }
            sfpi::dst_reg[0] = y;
        }
        sfpi::dst_reg++;
    }
}

// Initialize for add + rsqrt operation (just initializes rsqrt constants)
template <bool APPROXIMATION_MODE>
inline void init_add_rsqrt() {
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
