
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"

namespace ckernel::sfpu {

enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    RSUB = 4,
};  // BINOP_MODE

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_binop_with_scalar(std::uint32_t param) {
    const sfpi::vFloat parameter = Converter::as_float(param);

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP_MODE == ADD) {
            result = val + parameter;
        } else if constexpr (BINOP_MODE == SUB) {
            result = val - parameter;
        } else if constexpr (BINOP_MODE == MUL) {
            result = val * parameter;
        } else if constexpr (BINOP_MODE == DIV) {
            // inversion is carried out on host side and passed down
            result = val * parameter;
        } else if constexpr (BINOP_MODE == RSUB) {
            result = parameter - val;

            // This correction is added for logit(x) = log(x/(1-x)) since bf16 dest stores
            // truncate fp32->bf16 by default, but torch computes rsub result in bf16 with IEEE
            // round-to-nearest-even. The resulting small error is amplified by the log operation.
            if constexpr (!is_fp32_dest_acc_en) {
                result = float32_to_bf16_rne(result);
            }
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_add(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, ADD, ITERATIONS, is_fp32_dest_acc_en>(param);
}
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_sub(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, SUB, ITERATIONS, is_fp32_dest_acc_en>(param);
}
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_mul(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, MUL, ITERATIONS, is_fp32_dest_acc_en>(param);
}
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_div(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, DIV, ITERATIONS, is_fp32_dest_acc_en>(param);
}
template <bool APPROXIMATION_MODE, int ITERATIONS, bool is_fp32_dest_acc_en>
void calculate_rsub(std::uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, RSUB, ITERATIONS, is_fp32_dest_acc_en>(param);
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_add_int32(std::uint32_t scalar) {
    // dst (int32, 2's complement) + scalar
    const sfpi::vInt s = static_cast<int>(scalar);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a + s;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_sub_int32(std::uint32_t scalar) {
    // dst (int32, 2's complement) - scalar
    const sfpi::vInt s = static_cast<int>(scalar);
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vInt a = sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>();
        sfpi::dst_reg[0].mode<sfpi::DataLayout::I32>() = a - s;
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
