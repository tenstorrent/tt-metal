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
    bool SIGN_MAGNITUDE_FORMAT,
    InstrModLoadStore INSTRUCTION_MODE = InstrModLoadStore::INT32,
    int ITERATIONS = 8>
inline void calculate_add_int(const uint dst_offset) {
    _add_int_<APPROXIMATION_MODE, SIGN_MAGNITUDE_FORMAT, INSTRUCTION_MODE, ITERATIONS>(dst_offset);
}

}  // namespace sfpu
}  // namespace ckernel
