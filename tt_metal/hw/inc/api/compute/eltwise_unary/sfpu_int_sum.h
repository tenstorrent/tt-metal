// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_int_sum.h"
#endif

namespace ckernel {

ALWI void sfpu_sum_int_init() { MATH((sfpu::UnaryFn::init(sfpu::sum_int_init<APPROX>))); }

ALWI void sfpu_sum_int_col(std::uint32_t idst) {
    MATH((sfpu::SumIntCol<APPROX>::run<tensor_shape_from_tile_dims(16, 32)>(idst)));
}

ALWI void sfpu_sum_int_row(std::uint32_t idst) {
    MATH((sfpu::SumIntRow<APPROX>::run<tensor_shape_from_tile_dims(32, 16)>(idst)));
}

ALWI void sfpu_add_int(std::uint32_t idst, std::uint32_t dst_offset = 2, std::int32_t iterations = 8) {
    MATH((sfpu::AddIntDstOffset<APPROX>::run(idst, dst_offset)));
}

}  // namespace ckernel
