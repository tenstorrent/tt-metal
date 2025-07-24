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
    InstrModLoadStore INSTRUCTION_MODE = INT32,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_left_shift(const uint dst_offset) {
    _calculate_binary_left_shift_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(dst_offset);
}

template <
    bool APPROXIMATION_MODE,
    int ITERATIONS = 8,
    InstrModLoadStore INSTRUCTION_MODE = INT32,
    bool SIGN_MAGNITUDE_FORMAT = false>
inline void calculate_binary_right_shift(const uint dst_offset) {
    _calculate_binary_right_shift_<APPROXIMATION_MODE, ITERATIONS, INSTRUCTION_MODE, SIGN_MAGNITUDE_FORMAT>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel
