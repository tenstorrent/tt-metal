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

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_left_shift(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_binary_left_shift_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_right_shift(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    _calculate_binary_right_shift_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(
        dst_index_in0, dst_index_in1, dst_index_out);
}

}  // namespace sfpu
}  // namespace ckernel
