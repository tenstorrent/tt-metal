// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_assert.h"
#include "tensix_types.h"

namespace ckernel {

// The SFPU triangle solve is implemented on Blackhole only; these keep the compute API's signature on every
// architecture.
inline void llk_math_triangle_solve_sfpu_init() {
    LLK_ASSERT(false, "triangle_solve_tile is implemented on Blackhole only");
}

template <DataFormat L_FORMAT, bool L_NEGATED>
inline void llk_math_triangle_solve_sfpu_tile(
    [[maybe_unused]] const std::uint32_t l1_base,
    [[maybe_unused]] const std::uint32_t idst_in,
    [[maybe_unused]] const std::uint32_t idst_out) {
    LLK_ASSERT(false, "triangle_solve_tile is implemented on Blackhole only");
}

}  // namespace ckernel
