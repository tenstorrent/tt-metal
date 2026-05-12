// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 — compute API surface for add_binary_tile.

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_add_option4.h"
#include "llk_math_eltwise_binary_sfpu_macros_option4.h"
#endif

namespace ckernel {

using ckernel::isolate_sfpu::SfpuReg;

ALWI void add_binary_tile_init() { MATH(SFPU_BINARY_KERNEL_INIT(add)); }

// add_binary_tile.
//
// Template args (frequent → rare):
//   IN0_REG / IN1_REG / OUT_REG (default Dest) — per-operand register
//   ITERATIONS (default 4)                      — per-slice SFP iterations
//
// Runtime args:
//   idst0, idst1, odst   — dest tile indices (used when the corresponding
//                          slot maps to Dest; ignored when it maps to SrcS)
//   is_32bit_mode        — data-format-driven
//   vector_mode          — forward-compat (only RC supported today)
template <
    SfpuReg IN0_REG = SfpuReg::Dest,
    SfpuReg IN1_REG = SfpuReg::Dest,
    SfpuReg OUT_REG = SfpuReg::Dest,
    int ITERATIONS = 4>
ALWI void add_binary_tile(
    uint32_t idst0, uint32_t idst1, uint32_t odst, bool is_32bit_mode = false, int vector_mode = (int)VectorMode::RC) {
    MATH((SFPU_BINARY_KERNEL_V2(
        IN0_REG,
        IN1_REG,
        OUT_REG,
        &ckernel::sfpu::_calculate_add_<ITERATIONS>,
        idst0,
        idst1,
        odst,
        is_32bit_mode,
        vector_mode)));
}

}  // namespace ckernel
