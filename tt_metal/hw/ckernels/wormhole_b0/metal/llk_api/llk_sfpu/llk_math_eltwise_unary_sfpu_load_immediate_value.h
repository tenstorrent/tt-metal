// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_load_immediate_value.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

void llk_math_eltwise_unary_sfpu_load_imm(uint32_t dst_index, float val, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        [val]() { sfpu::load_immediate_value<APPROXIMATE>(val); }, dst_index, vector_mode);
}

void llk_math_eltwise_unary_sfpu_load_imm_init() {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_init<SfpuType::load_imm_value, APPROXIMATE>();
}

}  // namespace ckernel
