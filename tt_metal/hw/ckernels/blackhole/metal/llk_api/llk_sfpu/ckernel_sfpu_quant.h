// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_quant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _quant_int32_<APPROXIMATION_MODE, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_requant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _requant_int32_<APPROXIMATION_MODE, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8, bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_dequant_int32(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _dequant_int32_<APPROXIMATION_MODE, ITERATIONS, SIGN_MAGNITUDE_FORMAT>(dst_index_in0, dst_index_in1, dst_index_out);
}

template <bool APPROXIMATION_MODE>
void quant_init(const uint zero_point) {
    _init_quant_zero_point_<APPROXIMATION_MODE>(zero_point);
}

}  // namespace sfpu
}  // namespace ckernel
