// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"
// NOTE: The following header is from the external tt-llk and not defined in tt-metal
#include "sfpu/ckernel_sfpu_square.h"
#include "llk_defs.h"

namespace ckernel::sfpu {

template <ApproximationMode APPROX_MODE, int ITERATIONS = 8>
inline void calculate_square() {
    _calculate_square_<APPROX_MODE, ITERATIONS>();
}

}  // namespace ckernel::sfpu
