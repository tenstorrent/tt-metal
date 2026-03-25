// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "../../../../../../../tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_deepseek_top32_rm.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_deepseek_top32_rm_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::_top32_rm_init_);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_deepseek_top32_rm_local_sort(
    uint dst_index, int idir, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_bitonic_top32_phases_steps_<APPROXIMATE, is_fp32_dest_acc_en>, dst_index, vector_mode, idir);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool idir = false>
inline void llk_math_deepseek_top32_rm_merge(
    uint dst_index, bool across_tiles, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_bitonic_top32_merge_<APPROXIMATE, is_fp32_dest_acc_en, idir>,
        dst_index,
        vector_mode,
        across_tiles);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_deepseek_top32_rm_rebuild(
    uint dst_index, bool idir, bool skip_second, int vector_mode = (int)VectorMode::RC_custom) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::_bitonic_top32_rebuild_<APPROXIMATE, is_fp32_dest_acc_en>,
        dst_index,
        vector_mode,
        idir,
        skip_second);
}

}  // namespace ckernel
