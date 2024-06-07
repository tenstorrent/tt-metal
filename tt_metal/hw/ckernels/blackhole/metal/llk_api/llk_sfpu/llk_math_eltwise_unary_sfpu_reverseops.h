// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_reverseops.h"


namespace ckernel {

    /************** rsub ************/

    template <bool APPROXIMATE>
    inline void llk_math_eltwise_unary_sfpu_rsub_init() {
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::rsub_init<APPROXIMATE>);
    }

    template <bool APPROXIMATE>
    inline void llk_math_eltwise_unary_sfpu_rsub(uint dst_index, uint param0 = 0) {
        llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
            ckernel::sfpu::calculate_rsub<APPROXIMATE,8>,
            dst_index,
            (int)VectorMode::RC,
            param0);
    }

}
