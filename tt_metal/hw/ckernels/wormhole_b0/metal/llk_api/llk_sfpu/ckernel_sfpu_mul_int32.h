// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void _init_mul_int_() {}

template <bool APPROXIMATION_MODE>
inline void mul_int32_init() {}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void _mul_int_(const int, std::uint32_t, std::uint32_t) {}

template <bool APPROXIMATION_MODE>
inline void mul_int32(const int) {}

}  // namespace sfpu
}  // namespace ckernel
