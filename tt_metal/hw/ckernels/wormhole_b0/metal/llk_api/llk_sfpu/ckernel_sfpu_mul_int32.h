// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {}

template <bool APPROXIMATION_MODE>
inline void mul_int32(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out) {}

}  // namespace sfpu
}  // namespace ckernel
