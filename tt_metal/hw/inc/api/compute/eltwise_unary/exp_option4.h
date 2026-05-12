// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Option 4 v5 — compute API surface (sketch).
//
// Changes from v4:
//   - Template-arg order: IN_REG / OUT_REG before ITERATIONS, since users
//     override the register selection more often than the iteration count.
//     ITERATIONS sits at the end with default 4.
//   - vector_mode is threaded through as an opaque runtime arg, matching
//     the existing exp.h API. Currently RC-only; the inner kernel takes it
//     and ignores it for now.

#pragma once

#include "api/compute/common_globals.h"
#if defined(TRISC_MATH) || defined(TRISC_PACK)
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros_option4.h"
#endif

namespace ckernel {

using ckernel::isolate_sfpu::SfpuReg;

enum class InputClamping : uint8_t { ClampToNegative = 1, None = 0 };

template <
    bool approx = false,
    uint32_t scale = 0x3F800000,
    InputClamping input_clamping = InputClamping::ClampToNegative>
ALWI void exp_tile_init() {
    MATH(SFPU_TEMPLATE_INIT_KERNEL(
        exponential, sfpu::exp_init, approx, scale, (input_clamping == InputClamping::ClampToNegative)));
}

// exp_tile.
//
// Template-arg order (frequent → rare):
//   approx, scale_en, input_clamping  : op-specific
//   IN_REG  (default Dest)            : input register
//   OUT_REG (default Dest)            : output register
//   ITERATIONS (default 4)            : per-slice SFP iterations (1, 2, 4)
//
// Runtime args:
//   idst                              : tile index in dest section
//   is_32bit_mode (default false)     : data-format-driven; runtime in
//                                       high-level ops, constexpr at low level
//   vector_mode (default RC)          : forward-compat; only RC supported today
//   scale (default 1.0 in FP16b)      : op-specific runtime arg
//
// Usage:
//   exp_tile(0);                                                // Dest -> Dest, 16-bit
//   exp_tile(0, /*is_32bit_mode=*/true);                        // Dest -> Dest, 32-bit
//   exp_tile<..., SfpuReg::SrcS, SfpuReg::SrcS>(0);             // SrcS in-place
//   exp_tile<..., SfpuReg::SrcS, SfpuReg::Dest>(0);             // SrcS -> Dest
//   exp_tile<..., SfpuReg::Dest, SfpuReg::Dest, 1>(0);          // tiny tile, only first 2 rows
template <
    bool approx = false,
    bool scale_en = false,
    InputClamping input_clamping = InputClamping::ClampToNegative,
    SfpuReg IN_REG = SfpuReg::Dest,
    SfpuReg OUT_REG = SfpuReg::Dest,
    int ITERATIONS = 4>
ALWI void exp_tile(
    uint32_t idst,
    bool is_32bit_mode = false,
    int vector_mode = (int)VectorMode::RC,
    uint16_t scale = p_sfpu::kCONST_1_FP16B) {
    constexpr bool kClampNeg = (input_clamping == InputClamping::ClampToNegative);

    // The inner kernel signature takes (input_offset, output_offset,
    // vector_mode, scale). vector_mode is currently unused inside the kernel
    // but is part of the signature for forward compat with non-RC modes.
    MATH((SFPU_UNARY_KERNEL_V2(
        IN_REG,
        OUT_REG,
        ITERATIONS,
        &ckernel::sfpu::calculate_exponential<approx, DST_ACCUM_MODE, scale_en, ITERATIONS, kClampNeg>,
        idst,
        is_32bit_mode,
        vector_mode,
        scale)));
}

}  // namespace ckernel
