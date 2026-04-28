// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "sfpu/ckernel_sfpu_topk_xl.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::topk_xl_local_sort>(sfpu::_topk_xl_init_);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_local_sort(
    uint dst_index, bool ascending, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        sfpu::_topk_xl_local_sort_<APPROXIMATE>, dst_index, vector_mode, dst_index, ascending);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_merge(uint dst_index, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(sfpu::_topk_xl_merge_<APPROXIMATE>, dst_index, vector_mode, dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_xl_rebuild(
    uint dst_index, bool ascending, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_(
        sfpu::_topk_xl_rebuild_<APPROXIMATE>, dst_index, vector_mode, dst_index, ascending);
}

}  // namespace ckernel
