// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpu/ckernel_sfpu_sign.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_sign(std::uint32_t dst_index_in, std::uint32_t dst_index_out, const uint exponent_size_8) {
    _calculate_sign_<APPROXIMATION_MODE, ITERATIONS>(dst_index_in, dst_index_out, ITERATIONS, exponent_size_8);
}

}  // namespace sfpu
}  // namespace ckernel
