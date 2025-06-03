// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE, ckernel::sfpu::RelationalOp RELATIONAL_OP>
inline void llk_math_eltwise_binary_sfpu_relational_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(
        ckernel::sfpu::_sfpu_relational_int32_init_<APPROXIMATE, RELATIONAL_OP>);
}

template <bool APPROXIMATE, ckernel::sfpu::RelationalOp RELATIONAL_OP>
inline void llk_math_eltwise_binary_sfpu_relational_int32(
    uint dst_index0, uint32_t dst_index1, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::_calculate_sfpu_relational_int32_<APPROXIMATE, RELATIONAL_OP>,
        dst_index0,
        dst_index1,
        vector_mode);
}

}  // namespace ckernel
