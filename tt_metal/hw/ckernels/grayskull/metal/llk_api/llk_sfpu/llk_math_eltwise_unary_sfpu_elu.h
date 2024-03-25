// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_elu.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_elu_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::elu_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_elu(uint dst_index, uint param0 = 0) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>
                (ckernel::sfpu::calculate_elu<APPROXIMATE,4>,
				 ckernel::sfpu::calculate_elu<APPROXIMATE,4>,
				 dst_index, VectorMode::RC, param0);
}

}
