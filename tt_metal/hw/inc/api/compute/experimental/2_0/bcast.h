// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#include "experimental/2_0/llk_math_binary.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#include "experimental/2_0/llk_unpack_AB.h"
#endif

// =====================================================================================================
// Id-free (2.0) unary broadcast (unary_bcast): unpacks one L1 tile (with the requested broadcast mode) and
// datacopies it into DST. Takes an LLKOperand<Format,Shape> instead of a CB id: Format + Shape are
// compile-time NTTPs, l1_address is the only runtime state.
//
// A2D vs B2D selection:
//   * BroadcastType::NONE is a pass-through and uses A2D (reading a SrcB-resident tile back with B2D would
//     copy zeros and hang the unpacker).
//   * ROW / COL / SCALAR broadcasts leave the tile in SrcB and use B2D.
//   * 32-bit register formats (Float32/UInt32/Int32) always take the unpack-to-dest A2D path (SrcB is only
//     19 bits wide); decided from the DERIVED register format via infer_unpack_dst_format (fp32-dest-acc
//     rebias aware), so a 16-bit L1 format under fp32 dest-acc is still routed correctly.
// Blackhole only.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

namespace detail {
// Whether the unary broadcast takes the unpack-to-dest path, decided from the DERIVED REGISTER format
// (fp32-dest-acc rebias aware), not the raw L1 Format.
template <DataFormat Format>
constexpr bool unary_bcast_unpack_to_dest() {
    return is_32bit_format(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
}

// A2D/B2D data-copy direction for the unary broadcast, folded to a constant. Shared by unary_bcast_init /
// unary_bcast so the policy is spelled once.
template <BroadcastType bcast_type, DataFormat Format>
constexpr DataCopyType unary_bcast_dcopy() {
    return (unary_bcast_unpack_to_dest<Format>() || bcast_type == BroadcastType::NONE) ? DataCopyType::A2D
                                                                                       : DataCopyType::B2D;
}
}  // namespace detail

// clang-format off
/**
 * Paired init for unary_bcast. Configures the unpack + math pipeline for the given broadcast mode and
 * operand Format; call before unary_bcast. compute_kernel_hw_startup must already have run.
 *
 * | Param Type | Name      | Description                                                  | Type          | Valid Range           | Required |
 * |------------|-----------|--------------------------------------------------------------|---------------|-----------------------|----------|
 * | Template   | bcast_type| Broadcast mode (NONE pass-through / ROW / COL / SCALAR)       | BroadcastType | N/A                   | True     |
 * | Template   | Format    | Buffer L1 data format (deduced from the LLKOperand argument)  | DataFormat    | N/A                   | True     |
 * | Template   | Shape     | Tile geometry (deduced from the LLKOperand argument)         | TensorShape   | N/A                   | True     |
 * | Function   | src       | The source L1 operand (format + shape; address unused here)  | LLKOperand    | N/A                   | True     |
 */
// clang-format on
template <BroadcastType bcast_type, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_init(LLKOperand<Format, Shape> /*src*/) {
    static_assert(
        is_legal_tile_shape(Shape),
        "unary_bcast_init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    // 32-bit register formats use the unpack-to-dest A2D path (SrcB is only 19 bits wide); folds to a constant.
    constexpr bool enable_unpack_to_dest = detail::unary_bcast_unpack_to_dest<Format>();
    constexpr DataCopyType dcopy = detail::unary_bcast_dcopy<bcast_type, Format>();
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            bcast_type,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            enable_unpack_to_dest>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<Format, Shape>::descriptor,
          dcopy,
          DST_ACCUM_MODE,
          bcast_type>()));
}

// clang-format off
/**
 * Id-free unary broadcast. Unpacks one tile from the L1 region described by the LLKOperand (applying the
 * broadcast mode) and datacopies it into DST[dst_tile_index]. The DST register must be in the acquired
 * state. Blocking; compute-engine only. Pair with unary_bcast_init.
 *
 * | Param Type | Name           | Description                                                 | Type          | Valid Range | Required |
 * |------------|----------------|-------------------------------------------------------------|---------------|-------------|----------|
 * | Template   | bcast_type     | Broadcast mode (NONE pass-through / ROW / COL / SCALAR)      | BroadcastType | N/A         | True     |
 * | Template   | Format         | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat    | N/A         | True     |
 * | Template   | Shape          | Tile geometry (deduced from the LLKOperand argument)        | TensorShape   | N/A         | True     |
 * | Function   | src            | The source L1 operand (format + shape + address)            | LLKOperand    | N/A         | True     |
 * | Function   | dst_tile_index | Tile index in the DST register for the result               | uint32_t      | 0 to 15     | True     |
 */
// clang-format on
template <BroadcastType bcast_type, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    static_assert(
        is_legal_tile_shape(Shape),
        "unary_bcast: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    constexpr bool enable_unpack_to_dest = detail::unary_bcast_unpack_to_dest<Format>();
    constexpr DataCopyType dcopy = detail::unary_bcast_dcopy<bcast_type, Format>();
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            bcast_type,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            enable_unpack_to_dest>(src.l1_address)));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          dcopy,
          DST_ACCUM_MODE,
          bcast_type,
          enable_unpack_to_dest>(dst_tile_index)));
}

// clang-format off
/**
 * Paired uninit for unary_bcast. Restores the unpack + math pipeline; the operand is only used to select
 * the matching 32-bit unpack-to-dest uninit variant.
 *
 * | Param Type | Name      | Description                                                  | Type          | Valid Range | Required |
 * |------------|-----------|--------------------------------------------------------------|---------------|-------------|----------|
 * | Template   | bcast_type| Broadcast mode (must match the paired unary_bcast_init)      | BroadcastType | N/A         | True     |
 * | Template   | Format    | Buffer L1 data format (deduced from the LLKOperand argument)  | DataFormat    | N/A         | True     |
 * | Template   | Shape     | Tile geometry (deduced from the LLKOperand argument)         | TensorShape   | N/A         | True     |
 * | Function   | src       | The source L1 operand (format + shape; address unused here)  | LLKOperand    | N/A         | True     |
 */
// clang-format on
template <BroadcastType bcast_type, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_uninit(LLKOperand<Format, Shape> /*src*/) {
    constexpr bool enable_unpack_to_dest = detail::unary_bcast_unpack_to_dest<Format>();
    UNPACK((llk_unpack_A_uninit<bcast_type>()));
    MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, enable_unpack_to_dest>()));
}

// =====================================================================================================
// Id-free (2.0) binary broadcast (any_tiles_bcast<EltwiseBinaryType, BroadcastType>): C = A [op]
// broadcast(B), where B is a single tile broadcast across A per BroadcastType (ROW / COL / SCALAR). Takes
// one LLKOperand per input instead of CB ids: Format + Shape are compile-time NTTPs, l1_address the only
// runtime state. Format-free at the op level (src/dst register formats are programmed once at
// compute_kernel_hw_startup); geometry (for MATH + the AB-unpack init) comes from operand A.
//
// The ROW path additionally needs operand B's L1 format (forwarded to the unpacker); COL / SCALAR ignore
// it. BroadcastType::NONE (plain binary) belongs to experimental::add/sub/mul_tiles in eltwise_binary.h.
// =====================================================================================================

// clang-format off
/**
 * Paired init for any_tiles_bcast (generic op/dim). Configures MATH + the AB unpacker; call before
 * any_tiles_bcast. compute_kernel_hw_startup(a, b, out) must already have run. Geometry comes from operand
 * A (A.shape == B.shape assumed for the non-broadcast dims).
 *
 * | Param Type | Name      | Description                                                  | Type              | Valid Range | Required |
 * |------------|-----------|--------------------------------------------------------------|-------------------|-------------|----------|
 * | Template   | tBcastOp  | The binary op (ELWADD / ELWSUB / ELWMUL)                     | EltwiseBinaryType | N/A         | True     |
 * | Template   | tBcastDim | The broadcast dim (ROW / COL / SCALAR)                       | BroadcastType     | N/A         | True     |
 * | Template   | AFormat   | Operand A L1 data format (deduced from the LLKOperand)       | DataFormat        | N/A         | True     |
 * | Template   | AShape    | Operand A tile geometry (deduced from the LLKOperand)        | TensorShape       | N/A         | True     |
 * | Function   | a         | Operand A (drives geometry; address unused here)            | LLKOperand        | N/A         | True     |
 */
// clang-format on
template <EltwiseBinaryType tBcastOp, BroadcastType tBcastDim, DataFormat AFormat, TensorShape AShape>
ALWI void bcast_init(LLKOperand<AFormat, AShape> /*a*/) {
    static_assert(is_legal_tile_shape(AShape), "bcast_init: illegal tile shape for operand A.");
    static_assert(tBcastDim != BroadcastType::NONE, "bcast_init: use add/sub/mul_init for BroadcastType::NONE.");
    // ADD/SUB use LoFi (no fidelity multiplier), MUL uses MATH_FIDELITY; the ternary must stay inside MATH()
    // since MATH_FIDELITY is a MATH-thread-only macro.
    MATH((llk_math_eltwise_binary_init<
          LLKOperand<AFormat, AShape>::descriptor,
          tBcastOp,
          tBcastDim,
          (tBcastOp == EltwiseBinaryType::ELWMUL) ? MATH_FIDELITY : MathFidelity::LoFi>()));
    UNPACK((llk_unpack_AB_init<LLKOperand<AFormat, AShape>::descriptor, tBcastDim>(ckernel::Transpose::None)));
}

// clang-format off
/**
 * Binary broadcast: C = A [tBcastOp] broadcast(B) for one tile pair, writing DST[idst]. B is broadcast
 * across A per tBcastDim (COL: B's row 0; ROW: B's col 0; SCALAR: B[0,0]). Pair with bcast_init; DST must
 * be acquired. itile0/itile1 index within A/B; geometry comes from operand A. The ROW path forwards
 * operand B's L1 format to the unpacker.
 *
 * | Param Type | Name            | Description                                              | Type              | Valid Range | Required |
 * |------------|-----------------|--------------------------------------------------------|-------------------|-------------|----------|
 * | Template   | tBcastOp        | The binary op (ELWADD / ELWSUB / ELWMUL)               | EltwiseBinaryType | N/A         | True     |
 * | Template   | tBcastDim       | The broadcast dim (ROW / COL / SCALAR)                 | BroadcastType     | N/A         | True     |
 * | Template   | AFormat/AShape  | Operand A L1 format + geometry (deduced)               | DataFormat/TensorShape | N/A    | True     |
 * | Template   | BFormat/BShape  | Operand B L1 format + geometry (deduced)               | DataFormat/TensorShape | N/A    | True     |
 * | Function   | a / b           | Input operands (A -> SrcA, broadcast B -> SrcB)        | LLKOperand        | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B                              | uint32_t          | N/A         | True     |
 * | Function   | idst            | DST register index for the result                     | uint32_t          | 0 to 15     | True     |
 * | Function   | bcast_row_idx   | ROW broadcast: which row of B's tile to broadcast     | uint32_t          | N/A         | False    |
 */
// clang-format on
template <
    EltwiseBinaryType tBcastOp,
    BroadcastType tBcastDim,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void any_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    static_assert(is_legal_tile_shape(AShape), "any_tiles_bcast: illegal tile shape for operand A.");
    static_assert(is_legal_tile_shape(BShape), "any_tiles_bcast: illegal tile shape for operand B.");
    static_assert(tBcastDim != BroadcastType::NONE, "any_tiles_bcast: use add/sub/mul_tiles for BroadcastType::NONE.");
    // bcast_row_idx selects which row of B's tile the ROW broadcast reads; ignored by COL / SCALAR.
    // ADD/SUB use LoFi (no fidelity multiplier), MUL uses MATH_FIDELITY; the ternary must stay inside MATH()
    // since MATH_FIDELITY is a MATH-thread-only macro.
    MATH((llk_math_eltwise_binary<
          LLKOperand<AFormat, AShape>::descriptor,
          tBcastOp,
          tBcastDim,
          DST_ACCUM_MODE,
          (tBcastOp == EltwiseBinaryType::ELWMUL) ? MATH_FIDELITY : MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, tBcastDim, static_cast<std::uint8_t>(BFormat)>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1), bcast_row_idx)));
}

// clang-format off
/**
 * Id-free broadcast add: C = A + broadcast(B). Shorthand for any_tiles_bcast<ELWADD, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | 0 to 15     | True     |
 */
// clang-format on
template <BroadcastType tBcastDim, DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void add_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim>(a, b, itile0, itile1, idst, bcast_row_idx);
}

// clang-format off
/**
 * Id-free broadcast sub: C = A - broadcast(B). Shorthand for any_tiles_bcast<ELWSUB, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | 0 to 15     | True     |
 */
// clang-format on
template <BroadcastType tBcastDim, DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void sub_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim>(a, b, itile0, itile1, idst, bcast_row_idx);
}

// clang-format off
/**
 * Id-free broadcast mul: C = A * broadcast(B). Shorthand for any_tiles_bcast<ELWMUL, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics. The most common broadcast use.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | 0 to 15     | True     |
 */
// clang-format on
template <BroadcastType tBcastDim, DataFormat AFormat, TensorShape AShape, DataFormat BFormat, TensorShape BShape>
ALWI void mul_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim>(a, b, itile0, itile1, idst, bcast_row_idx);
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
