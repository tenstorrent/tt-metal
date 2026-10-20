// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Running per-element max + tile-column index for the batch-32 LM-head argmax (b32_argmax_compute.cpp): DST tile m
// holds the running maxima, tile i the tile column each came from; tile t is the next logits tile, column c. Strictly
// greater replaces, so a tie keeps the earlier column (the smaller vocabulary index).

#pragma once

#if defined(TRISC_MATH)

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu::b32am {
template <int ITERATIONS = 8>
inline void calculate_step(const uint m_idx, const uint t_idx, const uint i_idx, const uint c_bits) {
    constexpr uint tile = 32;  // SFPU rows per DST tile
    const sfpi::vFloat c = Converter::as_float(c_bits);
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat m = sfpi::dst_reg[m_idx * tile];
        const sfpi::vFloat t = sfpi::dst_reg[t_idx * tile];
        sfpi::vFloat i = sfpi::dst_reg[i_idx * tile];
        v_if(t > m) {
            m = t;
            i = c;
        }
        v_endif;
        sfpi::dst_reg[m_idx * tile] = m;
        sfpi::dst_reg[i_idx * tile] = i;
        sfpi::dst_reg++;
    }
}
}  // namespace ckernel::sfpu::b32am

namespace ckernel {
inline void b32_argmax_step_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(); }

// m (running max), t (new tile), i (running column index; passed as the "out" operand), c_bits: float bits of c
inline void b32_argmax_step(uint m, uint t, uint i, uint c_bits) {
    SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, b32am::calculate_step, (8), m, t, i, VectorMode::RC, c_bits);
}
}  // namespace ckernel

#endif  // TRISC_MATH
