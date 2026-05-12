// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — compute API surface for square_tile.
//
// Mirror of exp_option4.h with the op-specific knobs stripped down to what
// square actually needs (none — square has no approx, no scale, no clamp).
// This is the minimal shape of an Option 4 unary compute-API entry point.

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_square_option4.h"
#include "llk_math_eltwise_unary_sfpu_macros_option4.h"
#endif

namespace ckernel {

using ckernel::isolate_sfpu::SfpuReg;

ALWI void square_tile_init() { MATH(SFPU_UNARY_KERNEL_INIT(square)); }

// square_tile.
//
// Template-arg order (frequent → rare):
//   IN_REG  (default Dest)            : input register
//   OUT_REG (default Dest)            : output register
//   ITERATIONS (default 4)            : per-slice SFP iterations (1, 2, 4)
//
// Runtime args:
//   idst                              : tile index in dest section
//   is_32bit_mode (default false)     : data-format-driven; runtime in
//                                       high-level ops, constexpr at low level
//   vector_mode (default RC)          : forward-compat; only RC supported today
//
// Usage:
//   square_tile(0);                                          // Dest -> Dest, 16-bit
//   square_tile(0, /*is_32bit_mode=*/true);                  // Dest -> Dest, 32-bit
//   square_tile<SfpuReg::SrcS, SfpuReg::SrcS>(0);            // SrcS in-place
//   square_tile<SfpuReg::SrcS, SfpuReg::Dest>(0);            // SrcS -> Dest
//   square_tile<SfpuReg::Dest, SfpuReg::Dest, 1>(0);         // tiny tile: 2 rows / slice
template <SfpuReg IN_REG = SfpuReg::Dest, SfpuReg OUT_REG = SfpuReg::Dest, int ITERATIONS = 4>
ALWI void square_tile(uint32_t idst, bool is_32bit_mode = false, int vector_mode = (int)VectorMode::RC) {
    MATH((SFPU_UNARY_KERNEL_V2(
        IN_REG, OUT_REG, &ckernel::sfpu::_calculate_square_<ITERATIONS>, idst, is_32bit_mode, vector_mode)));
}

}  // namespace ckernel
