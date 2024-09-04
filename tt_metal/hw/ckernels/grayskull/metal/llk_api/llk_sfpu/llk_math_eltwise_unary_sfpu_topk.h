// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_topk.h"

namespace ckernel {

// New LLK SFPU APIs

// llk_math_eltwise_unary_sfpu_topk_init is unused for Grayskull
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::topk_init<APPROXIMATE>);
}

// llk_math_eltwise_unary_sfpu_topk_local_sort is unused for Grayskull
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_local_sort(uint dst_index, int idir, int i_end_phase, int i_start_phase,
                                                        int i_end_step, int i_start_step, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROXIMATE>,
        dst_index,
        vector_mode);
}

// llk_math_eltwise_unary_sfpu_topk_merge is unused for Grayskull
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_merge(uint dst_index, int m_iter, int k, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_merge<APPROXIMATE>,
        dst_index,
        vector_mode);
}

// llk_math_eltwise_unary_sfpu_topk_rebuild is unused for Grayskull
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_rebuild(uint dst_index, bool idir, int m_iter, int k, int logk,
                                                     int skip_second, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROXIMATE>,
        dst_index,
        vector_mode);
}

}
