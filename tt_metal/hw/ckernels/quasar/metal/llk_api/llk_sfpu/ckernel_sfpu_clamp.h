// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "sfpu/ckernel_sfpu_clamp.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = SFPU_ITERATIONS>
inline void calculate_clamp(std::uint32_t min_val, std::uint32_t max_val) {
    _calculate_clamp_<APPROXIMATION_MODE, ITERATIONS>(min_val, max_val);
}

}  // namespace sfpu
}  // namespace ckernel
