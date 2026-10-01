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

// Id-free (2.0) matmul: single-tile (matmul_init + matmul_tiles) and block (matmul_block_init + matmul_block).
// Two LLKOperands (in0, in1); in0 -> SrcB, in1 -> SrcA (matmul role swap), applied via
// compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out). Format-free at the op level -- these ops forward
// only geometry (via the descriptors) and per-tile L1 addresses; packing is separate (experimental::pack_tile).
// Throttle is fixed (MM_THROTTLE); dynamic throttle is not part of this surface.

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
    // Math execute takes no operand id; format-free.
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst)));
}

// clang-format off
/**
 * Short init for matmul_block. Configures the unpacker + math engine for block matmul. Pair with matmul_block.
 * compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out) must already have run. in0 -> SrcB, in1 -> SrcA.
 * ct_dim / rt_dim / kt_dim are the output-block geometry (see matmul_block) and are RUNTIME args; the register
 * formats + tile geometry are FORMAT-FREE and come from the two operand descriptors.
 *
 * | Function | in0 / in1 | Input operands (in0 -> SrcB, in1 -> SrcA) | LLKOperand | | True |
 * | Function | transpose | Transpose flag for tiles in B             | uint32_t   | | False |
 * | Function | ct_dim    | Output block column dim (== B col dim)    | uint32_t   | | False |
 * | Function | rt_dim    | Output block row dim (== A row dim)       | uint32_t   | | False |
 * | Function | kt_dim    | Inner dim (== A col dim)                  | uint32_t   | | False |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_block_init(
    LLKOperand<F0, S0> /*in0*/,
    LLKOperand<F1, S1> /*in1*/,
    std::uint32_t transpose = 0,
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1) {
    static_assert(is_legal_tile_shape(S0), "matmul_block_init: illegal tile shape for in0.");
    static_assert(is_legal_tile_shape(S1), "matmul_block_init: illegal tile shape for in1.");
    MATH((llk_math_matmul_init<
          LLKOperand<F0, S0>::descriptor,
          LLKOperand<F1, S1>::descriptor,
          MATH_FIDELITY,
          MM_THROTTLE>(transpose, ct_dim, rt_dim)));
    UNPACK((llk_unpack_AB_matmul_init<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        transpose, ct_dim, rt_dim, kt_dim)));
}

// clang-format off
/**
 * Block matmul C = A*B, accumulating into DST starting at idst. Pair with matmul_block_init. DST must be
 * acquired. A block is a rectangle of tiles: A is rt_dim x kt_dim, B is kt_dim x ct_dim, and the output C is
 * rt_dim x ct_dim tiles -- i.e. ct_dim*rt_dim output tiles produced in one call (kt_dim tiles along the shared
 * inner dim). The output must fit in DST. in0_tile_index / in1_tile_index are the base tile within in0 / in1;
 * the block strides across the operands from there. Geometry (partial_face + per-tile stride) comes from the
 * operand descriptors. transpose is programmed at matmul_block_init; the execute takes no transpose arg.
 *
 * | Function | in0 / in1                       | Input operands (in0 -> SrcB, in1 -> SrcA)    | LLKOperand | | True |
 * | Function | in0_tile_index / in1_tile_index | Base tile indices within in0 / in1           | uint32_t   | | True |
 * | Function | idst                            | DST register index for the first result tile | uint32_t   | | True |
 * | Function | ct_dim                          | Output block column dim (== B col dim)       | uint32_t   | | True |
 * | Function | rt_dim                          | Output block row dim (== A row dim)          | uint32_t   | | True |
 * | Function | kt_dim                          | Inner dim (== A col dim)                     | uint32_t   | | True |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_block(
    LLKOperand<F0, S0> in0,
    LLKOperand<F1, S1> in1,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst,
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim) {
    static_assert(is_legal_tile_shape(S0), "matmul_block: illegal tile shape for in0.");
    static_assert(is_legal_tile_shape(S1), "matmul_block: illegal tile shape for in1.");
    UNPACK((llk_unpack_AB_matmul<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        in0.l1_address, in1.l1_address, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    // Block execute takes no operand id; format-free, fixed MM_THROTTLE (no dynamic throttle).
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
}

}  // namespace experimental
}  // namespace ckernel
