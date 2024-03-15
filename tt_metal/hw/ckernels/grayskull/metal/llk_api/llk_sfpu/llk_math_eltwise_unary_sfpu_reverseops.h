// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_reverseops.h"


namespace ckernel {

    /************** rsub ************/

    template <bool APPROXIMATE>
    inline void llk_math_eltwise_unary_sfpu_rsub_init() {
        llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
    }

    template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
    inline void llk_math_eltwise_unary_sfpu_rsub(uint dst_index, uint param0 = 0) {
        llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE, Dst>
                                    (ckernel::sfpu::calculate_rsub<APPROXIMATE,4>,
                                    ckernel::sfpu::calculate_rsub<APPROXIMATE,4>,
                                    dst_index, VectorMode::RC, param0);
    }

}
