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
    if constexpr (APPROXIMATION_MODE || (ITERATIONS % 2) != 0) {
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
    } else {
        // Two vectors per step so that the refinement chains overlap; per lane the arithmetic is unchanged.
#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d += 2) {
            // Formed at every read of x: the body reads DEST again for its second step.
            const float addend = Converter::as_float(param0);
            auto operand = [addend](const sfpi::vFloat x) {
                if constexpr (INPUT_SCALE == 0x3f800000u) {
                    return x + addend;
                } else {
                    return x * Converter::as_float(INPUT_SCALE) + addend;
                }
            };
            sfpi::vFloat y0;
            sfpi::vFloat y1;
            _calculate_sqrt_body_accurate_x2_<true, FAST_APPROX>(
                [&] { return operand(sfpi::dst_reg[0]); }, [&] { return operand(sfpi::dst_reg[1]); }, y0, y1);

            if constexpr (!fp32_dest_acc_en && typed_bf16_store) {
                sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(y0, RoundMode::Nearest);
                sfpi::dst_reg[1] = sfpi::convert<sfpi::vFloat16b>(y1, RoundMode::Nearest);
            } else {
                if constexpr (!fp32_dest_acc_en) {
                    y0 = sfpi::convert<sfpi::vFloat16b>(y0, RoundMode::Nearest);
                    y1 = sfpi::convert<sfpi::vFloat16b>(y1, RoundMode::Nearest);
                }
                sfpi::dst_reg[0] = y0;
                sfpi::dst_reg[1] = y1;
            }
            sfpi::dst_reg += 2;
        }
    }
}

// Initialize for add + rsqrt operation (just initializes rsqrt constants)
template <bool APPROXIMATION_MODE>
inline void init_add_rsqrt() {
    sqrt_init<APPROXIMATION_MODE>();
}

}  // namespace ckernel::sfpu
