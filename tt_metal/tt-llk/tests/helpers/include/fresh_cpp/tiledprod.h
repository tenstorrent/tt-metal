// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Storm-contract semantic body for the `tiledprod-fresh` coverage row (metal
// ckernel_sfpu_tiled_prod.h calculate_tiled_prod, corpus manifest class
// D-ABSENT — zero dispatch anywhere).  Mathematical definition (running
// elementwise product down vector rows, the TTNN tiled_prod accumulation
// primitive): with r_k the k-th 32-lane vector row,
//   out_k = prod_{j<=k} r_j        for k = 0..ROWS-1
// rows beyond ROWS keep their input values.  The production contract
// processes ITERATIONS+1 = 9 rows per call (its documented off-by-one walk);
// ROWS mirrors that value so both arms cover identical rows.  The running
// product lives in fp32 registers (stores do not feed back); bf16 RNE store
// per the fresh float-body convention.
#include <cstdint>

namespace ckernel::sfpu
{

template <int ROWS>
__attribute__((noinline)) void calculate_tiled_prod_fresh_cpp()
{
    sfpi::vFloat run = 1.0f;
    for (int d = 0; d < ROWS; ++d)
    {
        run              = run * sfpi::dst_reg[0];
        sfpi::dst_reg[0] = sfpi::convert<sfpi::vFloat16b>(run, sfpi::RoundMode::Nearest);
        sfpi::dst_reg++;
    }
}

} // namespace ckernel::sfpu
