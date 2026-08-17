// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_int_sum.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

namespace ckernel {

ALWI void sfpu_sum_int_init() { SFPU(SFPU_UNARY_INIT_FN(unused, sfpu::sum_int_init, (APPROX))); }

ALWI void sfpu_sum_int_col(std::uint32_t idst) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sum_int_col, (APPROX), idst, VectorMode::R));
}

ALWI void sfpu_sum_int_row(std::uint32_t idst) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_sum_int_row, (APPROX), idst, VectorMode::C));
}

ALWI void sfpu_add_int(std::uint32_t idst, std::uint32_t dst_offset = 2, std::int32_t iterations = 8) {
    dest_order::touch_sfpu();
    SFPU(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, add_int, (APPROX, 8 /*ITERATIONS*/), idst, VectorMode::RC, dst_offset));
}

}  // namespace ckernel
