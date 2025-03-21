// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_sfpu_types.h"
#include "sfpi.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

namespace {
template <bool APPROXIMATE, int ITERATIONS = 8>
void _load_imm_(float imm) {
    vFloat const_v = s2vFloat16b(imm);
    for (int d = 0; d < ITERATIONS; d++) {
        dst_reg[0] = const_v;
        dst_reg++;
    }
}
}  // namespace

void llk_math_eltwise_unary_sfpu_load_imm(uint32_t dst_index, float val, int vector_mode = VectorMode::RC) {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>([val]() { _load_imm_<APPROXIMATE>(val); }, dst_index, vector_mode);
}

void llk_math_eltwise_unary_sfpu_load_imm_init() {
    constexpr bool APPROXIMATE = 0;
    llk_math_eltwise_unary_sfpu_init<SfpuType::load_imm_value, APPROXIMATE>();
}

}  // namespace sfpu
}  // namespace ckernel
