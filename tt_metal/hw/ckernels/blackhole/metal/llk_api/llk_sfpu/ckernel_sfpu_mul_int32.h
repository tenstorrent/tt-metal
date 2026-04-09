// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(uint32_t dst_index0, uint32_t dst_index1, uint32_t odst) {}

}  // namespace sfpu
}  // namespace ckernel
