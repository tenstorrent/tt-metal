// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_sfpu/ckernel_sfpu_alt_complex_rotate90.h"
#include "llk_sfpu/ckernel_sfpu_int_sum.h"
#include "llk_sfpu/ckernel_sfpu_tiled_prod.h"

namespace ckernel::sfpu
{

// These test-only adapters operate once on a complete 32x32 tile. Dispatch
// with VectorMode::None and YAML iterations: 32 (ITERATIONS == 8 after the
// existing Quasar generator's division by four), not the four-face RC mode.
template <bool APPROX, int ITERATIONS>
inline void calculate_parity_alt_complex_rotate90()
{
    static_assert(ITERATIONS == 8, "Structural parity requires one complete 32x32 tile.");
    // Each production iteration swaps two 32-lane vectors: 16 pairs per tile.
    calculate_alt_complex_rotate90<APPROX, 16>();
}

template <bool APPROX, int ITERATIONS>
inline void calculate_parity_int_sum_col()
{
    static_assert(ITERATIONS == 8, "Structural parity requires one complete 32x32 tile.");
    calculate_sum_int_col<APPROX>();
}

template <bool APPROX, int ITERATIONS>
inline void calculate_parity_int_sum_row()
{
    static_assert(ITERATIONS == 8, "Structural parity requires one complete 32x32 tile.");
    calculate_sum_int_row<APPROX>();
}

template <bool APPROX, int ITERATIONS>
inline void calculate_parity_tiled_prod()
{
    static_assert(ITERATIONS == 8, "Structural parity requires one complete 32x32 tile.");
    // The production helper handles ITERATIONS+1 vectors. 31 plus its final
    // vector therefore scans exactly one tile, without accessing a second tile.
    calculate_tiled_prod<APPROX, 31>();
}

} // namespace ckernel::sfpu
