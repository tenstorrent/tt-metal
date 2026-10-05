// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensix_types.h"

namespace ckernel {

// The SFPU triangle solve is implemented on Blackhole only. These keep the compute API's signature on every
// architecture; a call fails to compile here instead of silently leaving the output DST tile untouched.
template <DataFormat>
inline constexpr bool triangle_solve_supported_v = false;

inline void llk_math_triangle_solve_sfpu_init() {}

template <DataFormat L_FORMAT, bool L_NEGATED>
inline void llk_math_triangle_solve_sfpu_tile(
    [[maybe_unused]] const std::uint32_t l1_base,
    [[maybe_unused]] const std::uint32_t idst_in,
    [[maybe_unused]] const std::uint32_t idst_out) {
    static_assert(triangle_solve_supported_v<L_FORMAT>, "triangle_solve_tile is not implemented on Wormhole");
}

}  // namespace ckernel
