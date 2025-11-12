// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_logsigmoid.h"

namespace ckernel {

// LogSigmoid operation using pre-computed scaled input and exponential
template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_logsigmoid(
    uint dst_index_in0,  // Index for scaled input (beta * x)
    uint dst_index_in1,  // Index for exp result exp(-beta * x)
    uint dst_index_out,  // Index for output
    uint param0,         // beta (encoded as uint32_t)
    uint param1)         // threshold (encoded as uint32_t)
{
    _llk_math_eltwise_unary_sfpu_start_<SyncFull>(dst_index_out);

    sfpu::calculate_logsigmoid<APPROXIMATE, ITERATIONS>(dst_index_in0, dst_index_in1, dst_index_out, param0, param1);

    _llk_math_eltwise_unary_sfpu_done_();
}

// Initialize for logsigmoid operation
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_logsigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::logsigmoid_init<APPROXIMATE>);
}

}  // namespace ckernel
