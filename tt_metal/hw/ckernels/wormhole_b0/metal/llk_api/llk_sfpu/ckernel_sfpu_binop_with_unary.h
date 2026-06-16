
// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_conversions.h"
#include "sfpu/ckernel_sfpu_converter.h"
#include "sfpu/ckernel_sfpu_load_config.h"

namespace ckernel::sfpu {

enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3,
    RSUB = 4,
};  // BINOP_MODE

template <bool APPROXIMATION_MODE, int BINOP_MODE, int ITERATIONS = 8>
void calculate_binop_with_scalar(uint32_t param) {
    const sfpi::vFloat parameter = Converter::as_float(param);

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
            if constexpr (!DST_ACCUM_MODE) {
                result = float32_to_bf16_rne(result);
            }
        }

        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_add(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, ADD, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_sub(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, SUB, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_mul(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, MUL, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_div(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, DIV, ITERATIONS>(param);
    return;
}
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
void calculate_rsub(uint32_t param) {
    calculate_binop_with_scalar<APPROXIMATION_MODE, RSUB, ITERATIONS>(param);
    return;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_add_int32(uint32_t scalar) {
    int int_scalar = scalar;
    // Load value param to lreg2
    _sfpu_load_imm32_(p_sfpu::LREG2, int_scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);  // Using mov to preserve the scalar value after each iteration
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 4);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
void calculate_sub_int32(uint32_t scalar) {
    int int_scalar = scalar;
    // Load value scalar to lreg2
    _sfpu_load_imm32_(p_sfpu::LREG2, int_scalar);
    for (int d = 0; d < ITERATIONS; d++) {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        // Move scalar to lreg1 because lreg1 is the destination register in each loop iteration, so lreg2 keeps the
        // original scalar value.
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);
        // Used 6 as imod to convert operand B to 2's complement for sub operation
        TTI_SFPIADD(0, p_sfpu::LREG0, p_sfpu::LREG1, 6);
        TTI_SFPSTORE(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace ckernel::sfpu
