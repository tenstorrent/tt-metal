// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_matmul.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_AB_matmul.h"
#endif

#ifndef MM_THROTTLE
#define MM_THROTTLE 0
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// Id-free (2.0) single-tile matmul (matmul_init + matmul_tiles). Takes one LLKOperand per input; in0 -> SrcB
// and in1 -> SrcA (the matmul role swap). The role swap for the register formats is applied by
// compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) in the kernel; matmul is FORMAT-FREE at the op
// level, so these ops forward only geometry (via the descriptors) + the two per-tile L1 addresses. Packing is
// separate (experimental::pack_tile). Block matmul + dynamic throttle are not part of this Phase-1 surface.

// clang-format off
/**
 * Short init for matmul_tiles. Configures the unpacker + math engine for matmul. compute_kernel_hw_startup
 * <SrcOrder::Reverse>(in0, in1, out) must already have run. in0 -> SrcB, in1 -> SrcA.
 *
 * | Function | in0 / in1 | Input operands (in0 -> SrcB, in1 -> SrcA) | LLKOperand | | True |
 * | Function | transpose | Transpose flag for tiles in B             | uint32_t   | | False |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_init(LLKOperand<F0, S0> /*in0*/, LLKOperand<F1, S1> /*in1*/, std::uint32_t transpose = 0) {
    static_assert(is_legal_tile_shape(S0), "matmul_init: illegal tile shape for in0.");
    static_assert(is_legal_tile_shape(S1), "matmul_init: illegal tile shape for in1.");
    MATH((llk_math_matmul_init<
          LLKOperand<F0, S0>::descriptor,
          LLKOperand<F1, S1>::descriptor,
          MATH_FIDELITY,
          MM_THROTTLE>(transpose)));
    UNPACK((llk_unpack_AB_matmul_init<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(transpose)));
}

// clang-format off
/**
 * Tile matmul C = A*B, accumulating into DST[idst]. Pair with matmul_init. DST must be acquired. in0_tile_index
 * / in1_tile_index index within in0 / in1. Geometry (for MATH partial_face + unpack) comes from the operands.
 *
 * | Function | in0 / in1                     | Input operands                   | LLKOperand | | True |
 * | Function | in0_tile_index / in1_tile_index | Tile indices within in0 / in1  | uint32_t   | | True |
 * | Function | idst                          | DST register index for result C  | uint32_t   | | True |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_tiles(
    LLKOperand<F0, S0> in0,
    LLKOperand<F1, S1> in1,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(S0), "matmul_tiles: illegal tile shape for in0.");
    static_assert(is_legal_tile_shape(S1), "matmul_tiles: illegal tile shape for in1.");
    UNPACK((llk_unpack_AB_matmul<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        in0.l1_address, in1.l1_address, in0_tile_index, in1_tile_index)));
    // Matmul math execute takes no operand id -> reuse the legacy (format-free) execute directly.
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
