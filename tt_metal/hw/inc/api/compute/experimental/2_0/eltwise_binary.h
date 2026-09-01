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
#include "experimental/2_0/llk_unpack_A.h"  // dest-reuse path unpacks a single operand (A-only)
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// Id-free (2.0) two-operand eltwise binary (add/sub/mul). Each op takes one LLKOperand per input
// (A -> SrcA, B -> SrcB); format+geometry are NTTPs, L1 addresses the only runtime state. Format-free at
// the op level: src/dst register formats are programmed at compute_kernel_hw_startup; the op forwards only
// geometry (from operand A) + the two per-tile L1 addresses. Packing is separate (experimental::pack_tile).
// BroadcastType::NONE only (broadcast lives in bcast.h); dest-reuse (accumulation) variants are further down.

// clang-format off
/**
 * Paired init for a two-operand eltwise binary op. Configures MATH and the AB unpacker from operand A's
 * descriptor; geometry comes from operand A. compute_kernel_hw_startup(a_cb, b_cb, out_cb) must already have
 * programmed the formats. Operand B contributes nothing at init -- A.shape == B.shape is enforced at
 * execute (add/sub/mul_tiles).
 *
 * | Template | eltwise_binary_type | ELWADD / ELWSUB / ELWMUL               | EltwiseBinaryType |  | True |
 * | Function | a                   | Operand A (A -> SrcA); drives geometry | LLKOperand       |  | True |
 * | Function | acc_to_dest         | Accumulate the result into DST         | bool             |  | False |
 */
// clang-format on
template <EltwiseBinaryType eltwise_binary_type, DataFormat AFormat, TensorShape AShape>
ALWI void binary_tiles_init(LLKOperand<AFormat, AShape> /*a*/, bool acc_to_dest = false) {
    MATH((llk_math_eltwise_binary_init<
          LLKOperand<AFormat, AShape>::descriptor,
          eltwise_binary_type,
          BroadcastType::NONE,
          MATH_FIDELITY>(acc_to_dest)));
    UNPACK(
        (llk_unpack_AB_init<LLKOperand<AFormat, AShape>::descriptor, BroadcastType::NONE>(ckernel::Transpose::None)));
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
    binary_tiles_init<EltwiseBinaryType::ELWADD>(a, acc_to_dest);
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
    binary_tiles_init<EltwiseBinaryType::ELWSUB>(a, acc_to_dest);
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
    binary_tiles_init<EltwiseBinaryType::ELWMUL>(a, acc_to_dest);
}

// detail::tile_address (absolute per-tile L1 base) lives in llk_operand.h -- shared by every block op.

// clang-format off
/**
 * Element-wise C = A [op] B for one tile pair, writing DST[idst]. Pair with the matching *_init. DST must
 * be acquired. itile0/itile1 index within A/B; idst indexes DST. Geometry (for MATH) comes from operand A.
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode           | bool       | | False |
 * | Function | a / b           | Input operands                      | LLKOperand | | True |
 * | Function | itile0 / itile1 | Tile indices within A / B           | uint32_t   | | True |
 * | Function | idst            | DST register index for the result   | uint32_t   | | True |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
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
          is_fp32_dest_acc_en,
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
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode     | bool       |             | False    |
 * | Function   | a / b           | Input operands                    | LLKOperand | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B         | uint32_t   | N/A         | True     |
 * | Function   | idst            | DST register index for the result | uint32_t   | N/A         | True     |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
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
          is_fp32_dest_acc_en,
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
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode     | bool       |             | False    |
 * | Function   | a / b           | Input operands                    | LLKOperand | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B         | uint32_t   | N/A         | True     |
 * | Function   | idst            | DST register index for the result | uint32_t   | N/A         | True     |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
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
          is_fp32_dest_acc_en,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
}

// clang-format off
/**
 * Element-wise C = A + B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Loop form of
 * add_tiles (same semantics, same init: add_init). Tile (start_itile0 + i) of A pairs with tile
 * (start_itile1 + i) of B. DST must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to add          | uint32_t   | start_idst + ntiles <= 16 | True |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                  | bool       |             | False |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void add_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "add_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "add_block: operands A and B must have the same tile shape.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        add_tiles<is_fp32_dest_acc_en>(a, b, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Element-wise C = A - B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Loop form of
 * sub_tiles (same semantics, same init: sub_init). Tile (start_itile0 + i) of A pairs with tile
 * (start_itile1 + i) of B. DST must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to subtract     | uint32_t   | start_idst + ntiles <= 16 | True |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                  | bool       |             | False |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void sub_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "sub_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "sub_block: operands A and B must have the same tile shape.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        sub_tiles<is_fp32_dest_acc_en>(a, b, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// clang-format off
/**
 * Element-wise C = A * B over `ntiles` consecutive tile pairs, writing DST[start_idst + i]. Loop form of
 * mul_tiles (same semantics, same init: mul_init). Tile (start_itile0 + i) of A pairs with tile
 * (start_itile1 + i) of B. DST must be acquired; start_idst + ntiles <= DST size.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range | Required |
 * |------------|---------------|--------------------------------------------------|------------|-------------|----------|
 * | Function   | a / b         | Input operands (base address + geometry)         | LLKOperand | N/A         | True     |
 * | Function   | start_itile0  | Index of the first source tile within A          | uint32_t   | N/A         | True     |
 * | Function   | start_itile1  | Index of the first source tile within B          | uint32_t   | N/A         | True     |
 * | Function   | start_idst    | Index of the first destination tile in DST       | uint32_t   | 0 to 15     | True     |
 * | Function   | ntiles        | Number of consecutive tile pairs to multiply     | uint32_t   | start_idst + ntiles <= 16 | True |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                  | bool       |             | False |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void mul_block(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t start_itile0,
    std::uint32_t start_itile1,
    std::uint32_t start_idst,
    std::uint32_t ntiles) {
    static_assert(is_legal_tile_shape(AShape), "mul_block: illegal tile shape for operand A.");
    static_assert(same_tile_shape(AShape, BShape), "mul_block: operands A and B must have the same tile shape.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        mul_tiles<is_fp32_dest_acc_en>(a, b, start_itile0 + i, start_itile1 + i, start_idst + i);
    }
}

// =====================================================================================================================
// Dest-reuse (accumulation) binary ops: one operand is taken from the DST register instead of a second L1
// buffer, selected at runtime by dst_tile_index (so it carries no LLKOperand); only the other operand is an
// LLKOperand. Because a single operand is unpacked, this path uses llk_unpack_A (not llk_unpack_AB). BH
// accumulates the unpacked operand into DST at the unpacker (acc_to_dest = true).
// =====================================================================================================================

namespace detail {
// clang-format off
/**
 * (detail) Single source of truth for the id-free dest-reuse INIT. One source operand comes from DST, so only
 * the single L1 operand `in` is unpacked (llk_unpack_A). reuse_dest picks which source register the DST tile is
 * loaded into. compute_kernel_hw_startup(a, b, out) must already have programmed the formats.
 *
 * | Param Type | Name                | Description                                        | Type                       | Valid Range | Required |
 * |------------|---------------------|----------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | eltwise_binary_type | ELWADD / ELWSUB / ELWMUL                            | EltwiseBinaryType          | N/A         | True     |
 * | Template   | reuse_dest          | Which source register the DST operand loads into    | EltwiseBinaryReuseDestType | non-NONE    | True     |
 * | Function   | in                  | The single L1 operand (drives geometry + address)   | LLKOperand                 | N/A         | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType reuse_dest,
    DataFormat Format,
    TensorShape Shape>
ALWI void binary_reuse_dest_init(LLKOperand<Format, Shape> /*in*/) {
    static_assert(is_legal_tile_shape(Shape), "binary_reuse_dest_init: illegal tile shape for the L1 operand.");
    // BH: accumulate the unpacked operand into DST at the unpacker (acc_to_dest = true).
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            true /*acc_to_dest*/,
            reuse_dest>()));
    MATH((llk_math_eltwise_binary_init<
          LLKOperand<Format, Shape>::descriptor,
          eltwise_binary_type,
          BroadcastType::NONE,
          MATH_FIDELITY,
          reuse_dest>(0 /*acc_to_dest*/)));
}

// clang-format off
/**
 * (detail) Single source of truth for the id-free dest-reuse EXECUTE. The DST[dst_tile_index] tile is loaded
 * into SrcA (DEST_TO_SRCA) or SrcB (DEST_TO_SRCB); the op runs on SrcA & SrcB and writes back to
 * DST[dst_tile_index]. Assumes a prior op populated DST[dst_tile_index], else it reads zeroes.
 *
 * | Param Type | Name                | Description                                          | Type                       | Valid Range | Required |
 * |------------|---------------------|------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | eltwise_binary_type | ELWADD / ELWSUB / ELWMUL                              | EltwiseBinaryType          | N/A         | True     |
 * | Template   | reuse_dest          | Which source register the DST operand loads into      | EltwiseBinaryReuseDestType | non-NONE    | True     |
 * | Function   | in                  | The single L1 operand (base address + geometry)       | LLKOperand                 | N/A         | True     |
 * | Function   | in_tile_index       | Tile index within the L1 operand                      | uint32_t                   | N/A         | True     |
 * | Function   | dst_tile_index      | DST tile used as the other operand and as the result  | uint32_t                   | < DST size  | True     |
 */
// clang-format on
template <
    EltwiseBinaryType eltwise_binary_type,
    EltwiseBinaryReuseDestType reuse_dest,
    DataFormat Format,
    TensorShape Shape>
ALWI void binary_reuse_dest_tiles(
    LLKOperand<Format, Shape> in, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    static_assert(is_legal_tile_shape(Shape), "binary_reuse_dest_tiles: illegal tile shape for the L1 operand.");
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            true /*acc_to_dest*/,
            reuse_dest>(tile_address(in, in_tile_index))));
    MATH((llk_math_eltwise_binary<
          LLKOperand<Format, Shape>::descriptor,
          eltwise_binary_type,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          reuse_dest>(dst_tile_index, true /*clear_fp32_dst_acc*/)));
}
}  // namespace detail

// clang-format off
/**
 * Paired init for dest-reuse element-wise addition (add_reuse_dest_tiles<reuse_dest>). One addend is taken
 * from the DST register, so only the single L1 operand is unpacked; reuse_dest selects which source register
 * the DST tile loads into:
 *   - DEST_TO_SRCA: DST -> SrcA, in -> SrcB   (result = DST + in)
 *   - DEST_TO_SRCB: DST -> SrcB, in -> SrcA   (result = in + DST)
 * Pair with add_reuse_dest_tiles. compute_kernel_hw_startup(a, b, out) must already have run.
 *
 * | Param Type | Name       | Description                                                       | Type                       | Valid Range | Required |
 * |------------|------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)   | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in         | L1 operand unpacked into the source register not fed by DST        | LLKOperand                 | N/A         | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void add_reuse_dest_init(LLKOperand<Format, Shape> in) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call add_init(a).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWADD, reuse_dest>(in);
}

// clang-format off
/**
 * Paired init for dest-reuse element-wise subtraction (sub_reuse_dest_tiles<reuse_dest>). See
 * add_reuse_dest_init; DEST_TO_SRCA gives DST - in, DEST_TO_SRCB gives in - DST.
 *
 * | Param Type | Name       | Description                                                       | Type                       | Valid Range | Required |
 * |------------|------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)   | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in         | L1 operand unpacked into the source register not fed by DST        | LLKOperand                 | N/A         | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void sub_reuse_dest_init(LLKOperand<Format, Shape> in) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call sub_init(a).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWSUB, reuse_dest>(in);
}

// clang-format off
/**
 * Paired init for dest-reuse element-wise multiplication (mul_reuse_dest_tiles<reuse_dest>). See
 * add_reuse_dest_init; result = DST * in (DEST_TO_SRCA) or in * DST (DEST_TO_SRCB).
 *
 * | Param Type | Name       | Description                                                       | Type                       | Valid Range | Required |
 * |------------|------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest | Which source register the DST operand is loaded into (non-NONE)   | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in         | L1 operand unpacked into the source register not fed by DST        | LLKOperand                 | N/A         | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void mul_reuse_dest_init(LLKOperand<Format, Shape> in) {
    static_assert(
        reuse_dest != EltwiseBinaryReuseDestType::NONE,
        "reuse_dest must be DEST_TO_SRCA or DEST_TO_SRCB; for the two-operand op call mul_init(a).");
    detail::binary_reuse_dest_init<EltwiseBinaryType::ELWMUL, reuse_dest>(in);
}

// clang-format off
/**
 * Dest-reuse element-wise add: C = DST[dst_tile_index] + in, where one operand is the tile already in DST and
 * the other is unpacked from the L1 operand `in`. reuse_dest selects which source register the DST tile loads
 * into (DEST_TO_SRCA: DST->SrcA, in->SrcB; DEST_TO_SRCB: DST->SrcB, in->SrcA). Assumes a prior op populated
 * DST[dst_tile_index], else it reads zeroes. Pair with add_reuse_dest_init<reuse_dest>. The DST register must
 * be acquired.
 *
 * | Param Type | Name           | Description                                                        | Type                       | Valid Range | Required |
 * |------------|----------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest     | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in             | L1 operand unpacked into the source register not fed by DST         | LLKOperand                 | N/A         | True     |
 * | Function   | in_tile_index  | Index of the tile within the L1 operand                            | uint32_t                   | N/A         | True     |
 * | Function   | dst_tile_index | DST tile used as the other operand and as the result               | uint32_t                   | < DST size  | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void add_reuse_dest_tiles(
    LLKOperand<Format, Shape> in, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWADD, reuse_dest>(in, in_tile_index, dst_tile_index);
}

// clang-format off
/**
 * Dest-reuse element-wise subtract: C = DST[dst_tile_index] - in (DEST_TO_SRCA) or in - DST (DEST_TO_SRCB).
 * See add_reuse_dest_tiles; pair with sub_reuse_dest_init<reuse_dest>.
 *
 * | Param Type | Name           | Description                                                        | Type                       | Valid Range | Required |
 * |------------|----------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest     | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in             | L1 operand unpacked into the source register not fed by DST         | LLKOperand                 | N/A         | True     |
 * | Function   | in_tile_index  | Index of the tile within the L1 operand                            | uint32_t                   | N/A         | True     |
 * | Function   | dst_tile_index | DST tile used as the other operand and as the result               | uint32_t                   | < DST size  | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void sub_reuse_dest_tiles(
    LLKOperand<Format, Shape> in, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWSUB, reuse_dest>(in, in_tile_index, dst_tile_index);
}

// clang-format off
/**
 * Dest-reuse element-wise multiply: C = DST[dst_tile_index] * in (DEST_TO_SRCA) or in * DST (DEST_TO_SRCB).
 * See add_reuse_dest_tiles; pair with mul_reuse_dest_init<reuse_dest>.
 *
 * | Param Type | Name           | Description                                                        | Type                       | Valid Range | Required |
 * |------------|----------------|-------------------------------------------------------------------|----------------------------|-------------|----------|
 * | Template   | reuse_dest     | Which source register the DST operand is loaded into (non-NONE)    | EltwiseBinaryReuseDestType | N/A         | True     |
 * | Function   | in             | L1 operand unpacked into the source register not fed by DST         | LLKOperand                 | N/A         | True     |
 * | Function   | in_tile_index  | Index of the tile within the L1 operand                            | uint32_t                   | N/A         | True     |
 * | Function   | dst_tile_index | DST tile used as the other operand and as the result               | uint32_t                   | < DST size  | True     |
 */
// clang-format on
template <EltwiseBinaryReuseDestType reuse_dest, DataFormat Format, TensorShape Shape>
ALWI void mul_reuse_dest_tiles(
    LLKOperand<Format, Shape> in, std::uint32_t in_tile_index, std::uint32_t dst_tile_index) {
    detail::binary_reuse_dest_tiles<EltwiseBinaryType::ELWMUL, reuse_dest>(in, in_tile_index, dst_tile_index);
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
