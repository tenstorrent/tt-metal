// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/internal/op_hooks.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// Id-free (2.0) matmul, Quasar: same names and argument order as Blackhole (below). in0 -> SrcB, in1 -> SrcA.
// Quasar re-inits before every op: the inits program the operands' buffer descriptors, formats and MOPs.

// clang-format off
/**
 * Init for matmul_block, Quasar: C[rt_dim, ct_dim] += in0[rt_dim, kt_dim] x in1[kt_dim, ct_dim]. Unpack TRISC
 * programs Unp1 (in0 -> SrcB) and Unp0 (in1 -> SrcA) buffer descriptors at the operands' bases, their formats and
 * the matmul unpack MOP (with in1 transpose when set); math TRISC programs the ALU formats and the matmul MOP at the
 * kernel's math fidelity. Re-init when another op ran in between or the block shape changes. No uninit.
 *
 * | Param Type | Name      | Description                                    | Type       | Valid Range                     | Required |
 * |------------|-----------|------------------------------------------------|------------|---------------------------------|----------|
 * | Function   | in0 / in1 | Input operands (in0 -> SrcB, in1 -> SrcA)      | LLKOperand | Float16_b, Float16, Float32, MxFp8R, MxFp8P, MxInt8 | True     |
 * | Template   | ct_dim    | Output block columns (tiles)                   | uint32_t   | ct_dim * rt_dim <= section size | False    |
 * | Template   | rt_dim    | Output block rows (tiles)                      | uint32_t   | ct_dim * rt_dim <= section size | False    |
 * | Template   | kt_dim    | Inner dim (in0 columns, in1 rows) in tiles     | uint32_t   | >= 1                            | False    |
 * | Function   | transpose | Transpose each in1 tile (e.g. Q x K^T)         | uint32_t   | 0, 1 (1: full four-face in1)    | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void matmul_block_init(LLKOperand<F0, S0> in0, LLKOperand<F1, S1> in1, std::uint32_t transpose = 0) {
    static_assert(
        llk_2_0::is_fpu_operand_format(F0) && llk_2_0::is_fpu_operand_format(F1),
        "matmul_block_init: Float16_b / Float16 / Float32 / MxFp8R / MxFp8P / MxInt8 only.");
    static_assert(is_legal_tile_shape(S0) && is_legal_tile_shape(S1), "matmul_block_init: illegal tile shape.");
    static_assert(
        ct_dim * rt_dim <= detail::dest_section_tiles<DST_ACCUM_MODE>(), "matmul_block_init: block exceeds Dest");
    if (transpose) {
        UNPACK((llk_2_0::unpack_matmul_init<
                LLKOperand<F0, S0>::descriptor,
                LLKOperand<F1, S1>::descriptor,
                DST_ACCUM_MODE,
                true>(in0.l1_address, in1.l1_address, ct_dim, rt_dim, kt_dim)));
    } else {
        UNPACK((llk_2_0::unpack_matmul_init<
                LLKOperand<F0, S0>::descriptor,
                LLKOperand<F1, S1>::descriptor,
                DST_ACCUM_MODE,
                false>(in0.l1_address, in1.l1_address, ct_dim, rt_dim, kt_dim)));
    }
    MATH((llk_2_0::math_matmul_init<
          LLKOperand<F0, S0>::descriptor,
          LLKOperand<F1, S1>::descriptor,
          DST_ACCUM_MODE,
          MATH_FIDELITY>(ct_dim, rt_dim)));
}

// clang-format off
/**
 * Init for matmul_tiles: matmul_block_init<1, 1, 1>(in0, in1, transpose). Re-init when another op ran in between.
 * No uninit.
 *
 * | Param Type | Name      | Description                                    | Type       | Valid Range | Required |
 * |------------|-----------|------------------------------------------------|------------|-------------|----------|
 * | Function   | in0 / in1 | Input operands (in0 -> SrcB, in1 -> SrcA)      | LLKOperand | Float16_b, Float16, Float32, MxFp8R, MxFp8P, MxInt8 | True     |
 * | Function   | transpose | Transpose each in1 tile (e.g. Q x K^T)         | uint32_t   | 0, 1        | False    |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_init(LLKOperand<F0, S0> in0, LLKOperand<F1, S1> in1, std::uint32_t transpose = 0) {
    matmul_block_init<1, 1, 1>(in0, in1, transpose);
}

// clang-format off
/**
 * Tile matmul, Quasar: Dest[idst] += in0[in0_tile_index] x in1[in1_tile_index]. Tile i of an operand is L1 tile
 * entry_idx + i * stride_tiles of its buffer. To accumulate over k, call it kt times into the same idst. Pair with
 * matmul_init.
 *
 * | Param Type | Name                            | Description                     | Type       | Valid Range            | Required |
 * |------------|---------------------------------|---------------------------------|------------|------------------------|----------|
 * | Function   | in0 / in1                       | Input operands                  | LLKOperand | N/A                    | True     |
 * | Function   | in0_tile_index / in1_tile_index | Tile indices within in0 / in1   | uint32_t   | N/A                    | True     |
 * | Function   | idst                            | Dest tile index for the result  | uint32_t   | < Dest tiles / section | True     |
 */
// clang-format on
template <DataFormat F0, TensorShape S0, DataFormat F1, TensorShape S1>
ALWI void matmul_tiles(
    LLKOperand<F0, S0> in0,
    LLKOperand<F1, S1> in1,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst) {
    UNPACK((llk_2_0::unpack_matmul<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        in0.l1_address, in1.l1_address, in0.tile(in0_tile_index), in1.tile(in1_tile_index), 1, 1, 1)));
    MATH((llk_2_0::math_matmul_tile(idst)));
}

// clang-format off
/**
 * Block matmul step, Quasar: Dest[0, ct_dim * rt_dim) += in0 column block x in1 row block for one k: in0 tiles
 * in0_tile_index + r * kt_dim (r < rt_dim), in1 tiles in1_tile_index + c (c < ct_dim). For a full
 * C[rt, ct] = in0[rt, kt] x in1[kt, ct] call it for k < kt_dim with in0_tile_index = k, in1_tile_index = k * ct_dim
 * inside one section. Quasar: idst must be 0 (the block LLK writes from Dest tile 0; LLK_ASSERT). The tiles inside
 * a block step are contiguous, so both operands must have stride_tiles == 1 unless the block is 1x1 (asserted):
 * bind their DFBs BLOCKED or 1:1, since a multi-thread STRIDED binding gives stride_tiles > 1.
 * Pair with matmul_block_init of the same shape. No tails: the kernel picks a block that divides the output.
 *
 * | Param Type | Name                            | Description                               | Type       | Valid Range                     | Required |
 * |------------|---------------------------------|-------------------------------------------|------------|---------------------------------|----------|
 * | Template   | ct_dim / rt_dim                 | Output block columns / rows (tiles)       | uint32_t   | ct_dim * rt_dim <= section size | True     |
 * | Template   | kt_dim                          | Inner dim in tiles (in0 row pitch)        | uint32_t   | >= 1                            | True     |
 * | Function   | in0 / in1                       | Input operands                            | LLKOperand | N/A                             | True     |
 * | Function   | in0_tile_index / in1_tile_index | First tiles of this k step                | uint32_t   | N/A                             | True     |
 * | Function   | idst                            | Dest tile of the first result tile        | uint32_t   | 0                               | True     |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void matmul_block(
    LLKOperand<F0, S0> in0,
    LLKOperand<F1, S1> in1,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    [[maybe_unused]] std::uint32_t idst) {
    LLK_ASSERT(idst == 0, "matmul_block: Quasar block matmul writes from Dest tile 0");
    LLK_ASSERT(
        (ct_dim == 1 && rt_dim == 1) || (in0.stride_tiles == 1 && in1.stride_tiles == 1),
        "matmul_block: a block step needs contiguous operand tiles");
    UNPACK((llk_2_0::unpack_matmul<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        in0.l1_address, in1.l1_address, in0.tile(in0_tile_index), in1.tile(in1_tile_index), ct_dim, rt_dim, kt_dim)));
    MATH((llk_2_0::math_matmul_block(ct_dim, rt_dim)));
}

}  // namespace experimental
}  // namespace ckernel

#else  // Blackhole

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
 * ct_dim / rt_dim / kt_dim are the output-block geometry (see matmul_block), compile-time; the register
 * formats + tile geometry are FORMAT-FREE and come from the two operand descriptors.
 *
 * | Function | in0 / in1 | Input operands (in0 -> SrcB, in1 -> SrcA) | LLKOperand | | True |
 * | Template | ct_dim    | Output block column dim (== B col dim)    | uint32_t   | | False |
 * | Template | rt_dim    | Output block row dim (== A row dim)       | uint32_t   | | False |
 * | Template | kt_dim    | Inner dim (== A col dim)                  | uint32_t   | | False |
 * | Function | transpose | Transpose flag for tiles in B             | uint32_t   | | False |
 */
// clang-format on
template <
    std::uint32_t ct_dim = 1,
    std::uint32_t rt_dim = 1,
    std::uint32_t kt_dim = 1,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void matmul_block_init(LLKOperand<F0, S0> /*in0*/, LLKOperand<F1, S1> /*in1*/, std::uint32_t transpose = 0) {
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
 * | Template | ct_dim                          | Output block column dim (== B col dim)       | uint32_t   | | True |
 * | Template | rt_dim                          | Output block row dim (== A row dim)          | uint32_t   | | True |
 * | Template | kt_dim                          | Inner dim (== A col dim)                     | uint32_t   | | True |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    std::uint32_t kt_dim,
    DataFormat F0,
    TensorShape S0,
    DataFormat F1,
    TensorShape S1>
ALWI void matmul_block(
    LLKOperand<F0, S0> in0,
    LLKOperand<F1, S1> in1,
    std::uint32_t in0_tile_index,
    std::uint32_t in1_tile_index,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(S0), "matmul_block: illegal tile shape for in0.");
    static_assert(is_legal_tile_shape(S1), "matmul_block: illegal tile shape for in1.");
    UNPACK((llk_unpack_AB_matmul<LLKOperand<F0, S0>::descriptor, LLKOperand<F1, S1>::descriptor>(
        in0.l1_address, in1.l1_address, in0_tile_index, in1_tile_index, ct_dim, rt_dim, kt_dim)));
    // Block execute takes no operand id; format-free, fixed MM_THROTTLE (no dynamic throttle).
    MATH((llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(idst, ct_dim, rt_dim)));
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
