// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
// NOTE: The following header is from the external tt-llk dependency.
// It provides the implementation of `_calculate_square_`, which is used below.
// Any changes to the square kernel implementation should be made in tt-llk.
#include "sfpu/ckernel_sfpu_square.h"

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_square() {
    _calculate_square_<APPROXIMATION_MODE, ITERATIONS>();
}

}  // namespace ckernel::sfpu
