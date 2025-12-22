// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_silu.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_silu_init() {
<<<<<<< HEAD
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROXIMATE>(sfpu::silu_init);
=======
    llk_math_eltwise_unary_sfpu_init<SfpuType::silu, APPROXIMATE>(sfpu::silu_init<APPROXIMATE>);
>>>>>>> b47cee158f (silu fix)
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en>
inline void llk_math_eltwise_unary_sfpu_silu(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
<<<<<<< HEAD
        ckernel::sfpu::calculate_silu<is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
=======
        ckernel::sfpu::calculate_silu<APPROXIMATE, is_fp32_dest_acc_en, 8>, dst_index, vector_mode);
>>>>>>> b47cee158f (silu fix)
}

}  // namespace ckernel
