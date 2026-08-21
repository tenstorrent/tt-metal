// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_binary.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_AB.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// Id-free (2.0) two-operand eltwise binary (add/sub/mul). Each op takes one LLKOperand per input
// (A -> SrcA, B -> SrcB), format+geometry as NTTPs, L1 addresses the only runtime state. Binary is
// FORMAT-FREE at the op level: the src/dst register formats are programmed at compute_kernel_hw_startup;
// the op forwards only geometry (from operand A, mirroring legacy) + the two per-tile L1 addresses. Packing
// is done separately via the id-free experimental::pack_tile. The dest-reuse and broadcast variants are not
// part of this Phase-1 surface (BroadcastType::NONE only).

// clang-format off
/**
 * Paired init for a two-operand eltwise binary op. Configures MATH (always) and, when full_init, the AB
 * unpacker. Geometry is taken from operand A. compute_kernel_hw_startup(a_cb, b_cb, out_cb) must already
 * have programmed the formats.
 *
 * | Template | full_init           | Do the UNPACK init too (not just MATH) | bool             |  | True |
 * | Template | eltwise_binary_type | ELWADD / ELWSUB / ELWMUL               | EltwiseBinaryType |  | True |
 * | Function | a                   | Operand A (A -> SrcA); drives geometry | LLKOperand       |  | True |
 * | Function | acc_to_dest         | Accumulate the result into DST         | bool             |  | False |
 *
 * NOTE (PART B): the init forwards ONLY operand A's descriptor (geometry + format), mirroring legacy which
 * inits from a single operand. Operand B contributes nothing to init, so it is not taken here; the
 * A.shape == B.shape requirement is enforced at the execute (add/sub/mul_tiles), which does see both.
 */
// clang-format on
template <bool full_init, EltwiseBinaryType eltwise_binary_type, DataFormat AFormat, TensorShape AShape>
ALWI void binary_tiles_init(LLKOperand<AFormat, AShape> /*a*/, bool acc_to_dest = false) {
    MATH((llk_math_eltwise_binary_init<
          LLKOperand<AFormat, AShape>::descriptor,
          eltwise_binary_type,
          BroadcastType::NONE,
          MATH_FIDELITY>(acc_to_dest)));
    if constexpr (full_init) {
        UNPACK((llk_unpack_AB_init<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
            ckernel::Transpose::None)));
    }
}

// clang-format off
/**
 * Paired init for add_tiles (ELWADD). Configures MATH + the AB unpacker from operand A's descriptor;
 * compute_kernel_hw_startup(a, b, out) must already have programmed the formats. Only operand A is needed at
 * init (A.shape == B.shape is assumed, enforced at add_tiles) -- see binary_tiles_init.
 *
 * | Param Type | Name           | Description                              | Type                   | Valid Range | Required |
 * |------------|----------------|------------------------------------------|------------------------|-------------|----------|
 * | Template   | AFormat/AShape | Operand A L1 format + geometry (deduced) | DataFormat/TensorShape | N/A         | True     |
 * | Function   | a              | Operand A (drives geometry)              | LLKOperand             | N/A         | True     |
 * | Function   | acc_to_dest    | Accumulate the result into DST           | bool                   | N/A         | False    |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape>
ALWI void add_init(LLKOperand<AFormat, AShape> a, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(a, acc_to_dest);
}

// clang-format off
/**
 * Paired init for sub_tiles (ELWSUB). Configures MATH + the AB unpacker from operand A's descriptor;
 * compute_kernel_hw_startup(a, b, out) must already have programmed the formats. Only operand A is needed at
 * init (A.shape == B.shape is assumed, enforced at sub_tiles) -- see binary_tiles_init.
 *
 * | Param Type | Name           | Description                              | Type                   | Valid Range | Required |
 * |------------|----------------|------------------------------------------|------------------------|-------------|----------|
 * | Template   | AFormat/AShape | Operand A L1 format + geometry (deduced) | DataFormat/TensorShape | N/A         | True     |
 * | Function   | a              | Operand A (drives geometry)              | LLKOperand             | N/A         | True     |
 * | Function   | acc_to_dest    | Accumulate the result into DST           | bool                   | N/A         | False    |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape>
ALWI void sub_init(LLKOperand<AFormat, AShape> a, bool acc_to_dest = false) {
    binary_tiles_init<true, EltwiseBinaryType::ELWSUB>(a, acc_to_dest);
}

// clang-format off
/**
 * Paired init for mul_tiles (ELWMUL). Configures MATH + the AB unpacker from operand A's descriptor;
 * compute_kernel_hw_startup(a, b, out) must already have programmed the formats. Only operand A is needed at
 * init (A.shape == B.shape is assumed, enforced at mul_tiles) -- see binary_tiles_init.
 *
 * | Param Type | Name           | Description                              | Type                   | Valid Range | Required |
 * |------------|----------------|------------------------------------------|------------------------|-------------|----------|
 * | Template   | AFormat/AShape | Operand A L1 format + geometry (deduced) | DataFormat/TensorShape | N/A         | True     |
 * | Function   | a              | Operand A (drives geometry)              | LLKOperand             | N/A         | True     |
 * | Function   | acc_to_dest    | Accumulate the result into DST (mul defaults on) | bool           | N/A         | False    |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape>
ALWI void mul_init(LLKOperand<AFormat, AShape> a, bool acc_to_dest = true) {
    binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(a, acc_to_dest);
}

namespace detail {
// clang-format off
/**
 * (detail) Absolute per-tile L1 base for operand `op` at `tile_index` (16B words). The per-tile stride folds to
 * a compile-time constant from the operand geometry via tile_stride_words == one tile's L1 size (geometry-exact
 * for linear formats, exp section included for block floats), consistent with tilize/untilize/reduce/matmul.
 *
 * | Param Type | Name         | Description                            | Type                   | Valid Range | Required |
 * |------------|--------------|----------------------------------------|------------------------|-------------|----------|
 * | Template   | Format/Shape | Operand L1 format + geometry (deduced) | DataFormat/TensorShape | N/A         | True     |
 * | Function   | op           | The operand (base address + geometry)  | LLKOperand             | N/A         | True     |
 * | Function   | tile_index   | Tile index within the operand          | uint32_t               | N/A         | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI std::uint32_t tile_address(LLKOperand<Format, Shape> op, std::uint32_t tile_index) {
    constexpr std::uint32_t stride = tile_stride_words(static_cast<std::uint8_t>(Format), Shape);
    return op.l1_address + tile_index * stride;
}
}  // namespace detail

// clang-format off
/**
 * Element-wise C = A [op] B for one tile pair, writing DST[idst]. Pair with the matching *_init. The DST
 * register must be acquired. itile0/itile1 index within operand A/B; idst indexes DST. Geometry (for MATH)
 * comes from operand A, matching the legacy op.
 *
 * | Function | a / b           | Input operands                      | LLKOperand | | True |
 * | Function | itile0 / itile1 | Tile indices within A / B           | uint32_t   | | True |
 * | Function | idst            | DST register index for the result   | uint32_t   | | True |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void add_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(AShape), "add_tiles: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "add_tiles: operands A and B must have the same tile shape.");
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

// clang-format off
/**
 * Element-wise C = A - B for one tile pair, writing DST[idst]. Pair with sub_init. The DST register must be
 * acquired. itile0/itile1 index within operand A/B; idst indexes DST. Geometry (for MATH) comes from operand A.
 *
 * | Param Type | Name            | Description                       | Type       | Valid Range | Required |
 * |------------|-----------------|-----------------------------------|------------|-------------|----------|
 * | Function   | a / b           | Input operands                    | LLKOperand | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B         | uint32_t   | N/A         | True     |
 * | Function   | idst            | DST register index for the result | uint32_t   | N/A         | True     |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void sub_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(AShape), "sub_tiles: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "sub_tiles: operands A and B must have the same tile shape.");
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWSUB,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

// clang-format off
/**
 * Element-wise C = A * B for one tile pair, writing DST[idst]. Pair with mul_init. The DST register must be
 * acquired. itile0/itile1 index within operand A/B; idst indexes DST. Geometry (for MATH) comes from operand A.
 *
 * | Param Type | Name            | Description                       | Type       | Valid Range | Required |
 * |------------|-----------------|-----------------------------------|------------|-------------|----------|
 * | Function   | a / b           | Input operands                    | LLKOperand | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B         | uint32_t   | N/A         | True     |
 * | Function   | idst            | DST register index for the result | uint32_t   | N/A         | True     |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void mul_tiles(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst) {
    static_assert(is_legal_tile_shape(AShape), "mul_tiles: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "mul_tiles: operands A and B must have the same tile shape.");
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

// clang-format off
/**
 * Element-wise C = A + B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Block/loop form of
 * add_tiles: a pure compute-layer loop over the existing 2.0 add_tiles (mirrors copy_block over copy_tile), so
 * it inherits add_tiles's semantics and requires the same init (add_init). Tile (start_itile0 + i) of A is
 * paired with tile (start_itile1 + i) of B; each per-tile source address is the operand base offset by
 * (start_i + i) * tile_stride_words(Format, Shape) 16-byte words (folds to a compile-time stride). No CB id, no
 * register format on the API. The DST register must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to add          | uint32_t   | start_idst + ntiles <= 16 | True |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void add_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "add_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "add_block: operands A and B must have the same tile shape.");
    constexpr std::uint32_t stride_a = tile_stride_words(static_cast<std::uint8_t>(AFormat), AShape);
    constexpr std::uint32_t stride_b = tile_stride_words(static_cast<std::uint8_t>(BFormat), BShape);
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        add_tiles(
            LLKOperand<AFormat, AShape>(a.l1_address + (start_itile0 + i) * stride_a),
            LLKOperand<BFormat, BShape>(b.l1_address + (start_itile1 + i) * stride_b),
            0,
            0,
            start_idst + i);
    }
}

// clang-format off
/**
 * Element-wise C = A - B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Block/loop form of
 * sub_tiles: a pure compute-layer loop over the existing 2.0 sub_tiles (mirrors copy_block over copy_tile), so
 * it inherits sub_tiles's semantics and requires the same init (sub_init). Tile (start_itile0 + i) of A is
 * paired with tile (start_itile1 + i) of B; each per-tile source address is the operand base offset by
 * (start_i + i) * tile_stride_words(Format, Shape) 16-byte words (folds to a compile-time stride). No CB id, no
 * register format on the API. The DST register must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to subtract     | uint32_t   | start_idst + ntiles <= 16 | True |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void sub_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "sub_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "sub_block: operands A and B must have the same tile shape.");
    constexpr std::uint32_t stride_a = tile_stride_words(static_cast<std::uint8_t>(AFormat), AShape);
    constexpr std::uint32_t stride_b = tile_stride_words(static_cast<std::uint8_t>(BFormat), BShape);
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        sub_tiles(
            LLKOperand<AFormat, AShape>(a.l1_address + (start_itile0 + i) * stride_a),
            LLKOperand<BFormat, BShape>(b.l1_address + (start_itile1 + i) * stride_b),
            0,
            0,
            start_idst + i);
    }
}

// clang-format off
/**
 * Element-wise C = A * B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Block/loop form of
 * mul_tiles: a pure compute-layer loop over the existing 2.0 mul_tiles (mirrors copy_block over copy_tile), so
 * it inherits mul_tiles's semantics and requires the same init (mul_init). Tile (start_itile0 + i) of A is
 * paired with tile (start_itile1 + i) of B; each per-tile source address is the operand base offset by
 * (start_i + i) * tile_stride_words(Format, Shape) 16-byte words (folds to a compile-time stride). No CB id, no
 * register format on the API. The DST register must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to multiply     | uint32_t   | start_idst + ntiles <= 16 | True |
 */
// clang-format on
template <DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void mul_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "mul_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "mul_block: operands A and B must have the same tile shape.");
    constexpr std::uint32_t stride_a = tile_stride_words(static_cast<std::uint8_t>(AFormat), AShape);
    constexpr std::uint32_t stride_b = tile_stride_words(static_cast<std::uint8_t>(BFormat), BShape);
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        mul_tiles(
            LLKOperand<AFormat, AShape>(a.l1_address + (start_itile0 + i) * stride_a),
            LLKOperand<BFormat, BShape>(b.l1_address + (start_itile1 + i) * stride_b),
            0,
            0,
            start_idst + i);
    }
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
