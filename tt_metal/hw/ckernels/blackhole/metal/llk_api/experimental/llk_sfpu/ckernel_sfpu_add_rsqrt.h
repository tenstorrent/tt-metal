// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "ckernel.h"
#include "ckernel_sfpu_rsqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel::sfpu {

// Calculate: result = rsqrt(x + param0)
// param0 is the bit representation of a float
// This is useful for operations like RMSNorm: rsqrt(variance + epsilon)
// typed_bf16_store preserves the explicit BF16 destination-store mode used by
// fused normalization callers. Assigning the converted value back to vFloat
// instead retains the default source-format store selected by the SrcB format.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS,
    bool fp32_dest_acc_en,
    bool FAST_APPROX,
    bool typed_bf16_store = false>
inline void calculate_add_rsqrt(std::uint32_t param0) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat x = sfpi::dst_reg[0];
        sfpi::vFloat x_plus_addend = x + Converter::as_float(param0);

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

// Op class for rsqrt(x + addend), with addend passed as the bits of a float.
template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    bool fp32_dest_acc_en = false,
    bool FAST_APPROX = false,
    bool typed_bf16_store = false>
struct AddRsqrt
    : SfpuUnaryOp<AddRsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, typed_bf16_store>> {
    static inline __attribute__((always_inline)) void calculate(const std::uint32_t param0) {
        calculate_add_rsqrt<APPROXIMATION_MODE, ITERATIONS, fp32_dest_acc_en, FAST_APPROX, typed_bf16_store>(param0);
    }
    static inline __attribute__((always_inline)) void init_op() { init_add_rsqrt<APPROXIMATION_MODE>(); }
};

}  // namespace ckernel::sfpu
