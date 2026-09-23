// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_int_sum.h"
#endif

namespace ckernel {

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sum_int_init() {
    MATH((sfpu::SumInt<APPROX, true /*REDUCE_COLS*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sum_int_col(uint32_t idst) {
    MATH((sfpu::SumInt<APPROX, true /*REDUCE_COLS*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::R)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_sum_int_row(uint32_t idst) {
    MATH((sfpu::SumInt<APPROX, false /*REDUCE_COLS*/, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst, VectorMode::C)));
}

template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void sfpu_add_int(uint32_t idst, uint32_t dst_offset = 2, int32_t iterations = 8) {
    MATH((sfpu::SumIntAdd<APPROX, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(idst, VectorMode::RC, dst_offset)));
}

}  // namespace ckernel
