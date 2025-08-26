// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "llk_defs.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <
    ApproximationMode APPROX_MODE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_left_shift(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_binary_left_shift_<APPROX_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <
    ApproximationMode APPROX_MODE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_right_shift(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_binary_right_shift_<APPROX_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

}  // namespace sfpu
}  // namespace ckernel
