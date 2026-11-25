// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef ARCH_QUASAR

#include "api/compute/experimental/2_0/eltwise_binary.h"  // detail::binary_init / binary_one
#include "api/compute/experimental/2_0/internal/op_hooks.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Paired init for unary_bcast_src_to_dest_tile, Quasar (SrcB path; Quasar has no unpack-to-Dest broadcast). Unpack TRISC programs an Unp1 buffer descriptor at src's base and the
 * unpack-to-SrcB broadcast MOP; math TRISC programs the SrcB -> Dest broadcast datacopy. Re-init when another op
 * ran in between (Quasar rule). No uninit. Float16_b / Float16, full 32x32 tiles, ROW / COL / SCALAR.
 *
 * | Param Type | Name                | Description                              | Type          | Valid Range        | Required |
 * |------------|---------------------|------------------------------------------|---------------|--------------------|----------|
 * | Template   | bcast_type          | Broadcast mode (ROW / COL / SCALAR)      | BroadcastType | not NONE           | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                | bool          | DST_ACCUM_MODE     | False    |
 * | Function   | src                 | Source operand (e.g. DataflowBuffer::front) | LLKOperand | N/A                | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void unary_bcast_src_to_dest_init(LLKOperand<Format, Shape> src) {
    static_assert(
        bcast_type != BroadcastType::NONE,
        "unary_bcast_src_to_dest_init (Quasar 2.0): use copy_src_to_dest_init for NONE.");
    static_assert(
        llk_2_0::is_supported_format(Format),
        "unary_bcast_src_to_dest_init (Quasar 2.0): Float16_b / Float16 only (no unary_bcast_direct_to_dest on "
        "Quasar).");
    static_assert(
        Shape.face_r_dim == MAX_FACE_R_DIM && Shape.num_faces_r_dim == MAX_NUM_FACES_R_DIM &&
            Shape.num_faces_c_dim == MAX_NUM_FACES_C_DIM,
        "unary_bcast_src_to_dest_init (Quasar 2.0): 32x32 tiles only.");
    static_assert(
        is_fp32_dest_acc_en == DST_ACCUM_MODE,
        "unary_bcast_src_to_dest_init: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    constexpr LLKMemDescriptor D = LLKOperand<Format, Shape>::descriptor;
    UNPACK((llk_2_0::unpack_B_bcast_init<D, bcast_type, is_fp32_dest_acc_en>(src.l1_address)));
    MATH((llk_2_0::math_unary_bcast_init<D, bcast_type, is_fp32_dest_acc_en>()));
}

// clang-format off
/**
 * Id-free unary broadcast of src's first tile into Dest tile `idst` through SrcB, Quasar. Pair with
 * unary_bcast_src_to_dest_init. Call between tile_regs_acquire and tile_regs_commit.
 *
 * | Param Type | Name                | Description                                  | Type          | Valid Range            | Required |
 * |------------|---------------------|----------------------------------------------|---------------|------------------------|----------|
 * | Template   | bcast_type          | Broadcast mode (as at init)                  | BroadcastType | ROW / COL / SCALAR     | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                    | bool          | DST_ACCUM_MODE         | False    |
 * | Function   | src                 | Source operand (its base = the init's)       | LLKOperand    | N/A                    | True     |
 * | Function   | dst_tile_index      | Dest tile index for the result               | uint32_t      | < Dest tiles / section | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
inline void unary_bcast_src_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    UNPACK((llk_2_0::unpack_B_bcast(src.l1_address, src.tile(0))));
    MATH((llk_2_0::math_unary_bcast(dst_tile_index)));
}

// clang-format off
/**
 * Paired init for any_tiles_bcast, Quasar. Like add/sub/mul_init, but operand B is broadcast per tBcastDim. Both
 * operands are needed: the init programs both buffer descriptors. Re-init when another op ran in between (Quasar
 * rule). No uninit. Float16_b / Float16; ROW / COL / SCALAR.
 *
 * | Param Type | Name      | Description                                   | Type              | Valid Range | Required |
 * |------------|-----------|-----------------------------------------------|-------------------|-------------|----------|
 * | Template   | tBcastOp  | The binary op (ELWADD / ELWSUB / ELWMUL)      | EltwiseBinaryType | N/A         | True     |
 * | Template   | tBcastDim | The broadcast dim (ROW / COL / SCALAR)        | BroadcastType     | not NONE    | True     |
 * | Function   | a / b     | Input operands (A -> SrcA, broadcast B -> SrcB) | LLKOperand      | N/A         | True     |
 */
// clang-format on
template <
    EltwiseBinaryType tBcastOp,
    BroadcastType tBcastDim,
    DataFormat FA,
    TensorShape SA,
    DataFormat FB,
    TensorShape SB>
inline void bcast_init(LLKOperand<FA, SA> a, LLKOperand<FB, SB> b) {
    static_assert(tBcastDim != BroadcastType::NONE, "bcast_init: use add/sub/mul_init for BroadcastType::NONE.");
    detail::binary_init<tBcastOp, DST_ACCUM_MODE, tBcastDim>(a, b, false);
}

// clang-format off
/**
 * Binary broadcast, Quasar (Blackhole's form): Dest[idst] = A[itile0] [tBcastOp] broadcast(B[itile1]). Pair with
 * bcast_init. Call between tile_regs_acquire and tile_regs_commit.
 *
 * | Param Type | Name                | Description                                | Type              | Valid Range            | Required |
 * |------------|---------------------|--------------------------------------------|-------------------|------------------------|----------|
 * | Template   | tBcastOp            | The binary op (as at init)                 | EltwiseBinaryType | N/A                    | True     |
 * | Template   | tBcastDim           | The broadcast dim (as at init)             | BroadcastType     | ROW / COL / SCALAR     | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                  | bool              | DST_ACCUM_MODE         | False    |
 * | Function   | a / b               | Input operands                             | LLKOperand        | N/A                    | True     |
 * | Function   | itile0 / itile1     | Tile indices within A / B                  | uint32_t          | N/A                    | True     |
 * | Function   | idst                | Dest tile index for the result             | uint32_t          | < Dest tiles / section | True     |
 * | Function   | bcast_row_idx       | ROW broadcast source row                   | uint32_t          | 0 on Quasar            | False    |
 */
// clang-format on
template <
    EltwiseBinaryType tBcastOp,
    BroadcastType tBcastDim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat FA,
    TensorShape SA,
    DataFormat FB,
    TensorShape SB>
inline void any_tiles_bcast(
    LLKOperand<FA, SA> a,
    LLKOperand<FB, SB> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    [[maybe_unused]] std::uint32_t bcast_row_idx = 0) {
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "any_tiles_bcast: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    LLK_ASSERT(bcast_row_idx == 0, "any_tiles_bcast: Quasar broadcasts row 0 only");
    detail::binary_one<tBcastOp, tBcastDim>(a, b, itile0, itile1, idst);
}

}  // namespace experimental
}  // namespace ckernel

#else  // Blackhole

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#include "experimental/2_0/llk_math_binary.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#include "experimental/2_0/llk_unpack_AB.h"
#endif

// =====================================================================================================
// Id-free (2.0) unary broadcast: unpacks one L1 tile (with the requested broadcast mode) and datacopies it into
// DST. Two APIs, the caller picks the path:
//   * unary_bcast_src_to_dest_*: through a Src register. ROW / COL / SCALAR leave the tile in SrcB (B2D);
//     NONE is a pass-through through SrcA (A2D). 16/19-bit register formats only.
//   * unary_bcast_direct_to_dest_*: unpack straight to DST (A2D). 32-bit register formats only (Float32 at fp32
//     Dest, Int32, UInt32), which do not fit the 19-bit Src registers.
// Blackhole section; Quasar is above.
// =====================================================================================================

namespace ckernel {
namespace experimental {

namespace detail {
// Shared body of the two unary broadcast paths; `direct` = unpack-to-Dest.
template <bool direct, BroadcastType bcast_type, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_init() {
    static_assert(
        is_legal_tile_shape(Shape),
        "unary_bcast init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        direct || !is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "unary_bcast_src_to_dest: a 32-bit register format does not fit the Src registers; use "
        "unary_bcast_direct_to_dest_*.");
    static_assert(
        !direct || is_unpack_to_dest<Format, is_fp32_dest_acc_en>(),
        "unary_bcast_direct_to_dest: unpack-to-Dest takes only a 32-bit register format (Float32 at fp32 Dest, "
        "Int32, UInt32); use unary_bcast_src_to_dest_*.");
    constexpr DataCopyType dcopy =
        (direct || bcast_type == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            bcast_type,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            direct>()));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<Format, Shape>::descriptor,
          dcopy,
          is_fp32_dest_acc_en,
          bcast_type>()));
}

template <bool direct, BroadcastType bcast_type, bool is_fp32_dest_acc_en, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    constexpr DataCopyType dcopy =
        (direct || bcast_type == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            is_fp32_dest_acc_en,
            bcast_type,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            direct>(src.l1_address)));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          dcopy,
          is_fp32_dest_acc_en,
          bcast_type,
          direct>(dst_tile_index)));
}
}  // namespace detail

// clang-format off
/**
 * Paired init for unary_bcast_src_to_dest_tile: the broadcast goes through a Src register (SrcB for ROW / COL /
 * SCALAR, SrcA for the NONE pass-through). 16/19-bit register formats only; a 32-bit register format is rejected
 * at compile time: use unary_bcast_direct_to_dest_init. compute_kernel_hw_startup must already have run.
 *
 * | Param Type | Name                | Description                                                  | Type          | Valid Range | Required |
 * |------------|---------------------|--------------------------------------------------------------|---------------|-------------|----------|
 * | Template   | bcast_type          | Broadcast mode (NONE pass-through / ROW / COL / SCALAR)      | BroadcastType | N/A         | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                                    | bool          | N/A         | False    |
 * | Function   | src                 | The source L1 operand (format + shape; address unused here)  | LLKOperand    | N/A         | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_src_to_dest_init(LLKOperand<Format, Shape> /*src*/) {
    detail::unary_bcast_init<false, bcast_type, is_fp32_dest_acc_en, Format, Shape>();
}

// clang-format off
/**
 * Paired init for unary_bcast_direct_to_dest_tile: the tile is unpacked straight to DST (A2D). Blackhole only.
 * 32-bit register formats only (Float32 at fp32 Dest, Int32, UInt32); anything else is rejected at compile time:
 * use unary_bcast_src_to_dest_init. compute_kernel_hw_startup must already have run.
 *
 * | Param Type | Name                | Description                                                  | Type          | Valid Range | Required |
 * |------------|---------------------|--------------------------------------------------------------|---------------|-------------|----------|
 * | Template   | bcast_type          | Broadcast mode (NONE pass-through / ROW / COL / SCALAR)      | BroadcastType | N/A         | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                                    | bool          | N/A         | False    |
 * | Function   | src                 | The source L1 operand (format + shape; address unused here)  | LLKOperand    | N/A         | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_direct_to_dest_init(LLKOperand<Format, Shape> /*src*/) {
    detail::unary_bcast_init<true, bcast_type, is_fp32_dest_acc_en, Format, Shape>();
}

// clang-format off
/**
 * Id-free unary broadcast through a Src register: unpacks one tile from src (applying the broadcast mode) and
 * datacopies it into DST[dst_tile_index]. DST must be acquired. Pair with unary_bcast_src_to_dest_init.
 *
 * | Param Type | Name                | Description                                     | Type          | Valid Range                                    | Required |
 * |------------|---------------------|-------------------------------------------------|---------------|------------------------------------------------|----------|
 * | Template   | bcast_type          | Broadcast mode (as at init)                     | BroadcastType | N/A                                            | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                       | bool          | N/A                                            | False    |
 * | Function   | src                 | The source L1 operand (format + shape + address) | LLKOperand   | N/A                                            | True     |
 * | Function   | dst_tile_index      | Tile index in DST for the result                | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_src_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    detail::unary_bcast_tile<false, bcast_type, is_fp32_dest_acc_en>(src, dst_tile_index);
}

// clang-format off
/**
 * Id-free unary broadcast unpacked straight to DST: unpacks one tile from src (applying the broadcast mode) into
 * DST[dst_tile_index]. Blackhole only. DST must be acquired. Pair with unary_bcast_direct_to_dest_init.
 *
 * | Param Type | Name                | Description                                     | Type          | Valid Range                                    | Required |
 * |------------|---------------------|-------------------------------------------------|---------------|------------------------------------------------|----------|
 * | Template   | bcast_type          | Broadcast mode (as at init)                     | BroadcastType | N/A                                            | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                       | bool          | N/A                                            | False    |
 * | Function   | src                 | The source L1 operand (format + shape + address) | LLKOperand   | N/A                                            | True     |
 * | Function   | dst_tile_index      | Tile index in DST for the result                | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <BroadcastType bcast_type, bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void unary_bcast_direct_to_dest_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    detail::unary_bcast_tile<true, bcast_type, is_fp32_dest_acc_en>(src, dst_tile_index);
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
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                          | bool              | N/A         | False    |
 * | Template   | AFormat/AShape  | Operand A L1 format + geometry (deduced)               | DataFormat/TensorShape | N/A    | True     |
 * | Template   | BFormat/BShape  | Operand B L1 format + geometry (deduced)               | DataFormat/TensorShape | N/A    | True     |
 * | Function   | a / b           | Input operands (A -> SrcA, broadcast B -> SrcB)        | LLKOperand        | N/A         | True     |
 * | Function   | itile0 / itile1 | Tile indices within A / B                              | uint32_t          | N/A         | True     |
 * | Function   | idst            | DST register index for the result                     | uint32_t          | Must be less than the acquired size of DST REG | True     |
 * | Function   | bcast_row_idx   | ROW broadcast: which row of B's tile to broadcast     | uint32_t          | N/A         | False    |
 */
// clang-format on
template <
    EltwiseBinaryType tBcastOp,
    BroadcastType tBcastDim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
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
          is_fp32_dest_acc_en,
          (tBcastOp == EltwiseBinaryType::ELWMUL) ? MATH_FIDELITY : MathFidelity::LoFi,
          EltwiseBinaryReuseDestType::NONE>(idst, true /*clear_fp32_dst_acc*/)));
    UNPACK((llk_unpack_AB<LLKOperand<AFormat, AShape>::descriptor, tBcastDim, static_cast<std::uint8_t>(BFormat)>(
        detail::tile_address(a, itile0), detail::tile_address(b, itile1), bcast_row_idx)));
}

// clang-format off
/**
 * Two-operand form of bcast_init, matching Quasar's bcast_init(a, b) so one kernel source builds on both. Blackhole
 * needs only operand A at init; B is ignored.
 *
 * | Param Type | Name      | Description                              | Type              | Valid Range | Required |
 * |------------|-----------|------------------------------------------|-------------------|-------------|----------|
 * | Template   | tBcastOp  | The binary op (ELWADD / ELWSUB / ELWMUL) | EltwiseBinaryType | N/A         | True     |
 * | Template   | tBcastDim | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType     | not NONE    | True     |
 * | Function   | a         | Operand A (drives geometry)              | LLKOperand        | N/A         | True     |
 * | Function   | b         | Operand B (unused on Blackhole)          | LLKOperand        | N/A         | True     |
 */
// clang-format on
template <
    EltwiseBinaryType tBcastOp,
    BroadcastType tBcastDim,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void bcast_init(LLKOperand<AFormat, AShape> a, LLKOperand<BFormat, BShape> /*b*/) {
    bcast_init<tBcastOp, tBcastDim>(a);
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR

// Broadcast shorthands, shared by both arches.
namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free broadcast add: C = A + broadcast(B). Shorthand for any_tiles_bcast<ELWADD, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode           | bool          | N/A         | False    |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    BroadcastType tBcastDim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void add_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWADD, tBcastDim, is_fp32_dest_acc_en>(
        a, b, itile0, itile1, idst, bcast_row_idx);
}

// clang-format off
/**
 * Id-free broadcast sub: C = A - broadcast(B). Shorthand for any_tiles_bcast<ELWSUB, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode           | bool          | N/A         | False    |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    BroadcastType tBcastDim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void sub_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWSUB, tBcastDim, is_fp32_dest_acc_en>(
        a, b, itile0, itile1, idst, bcast_row_idx);
}

// clang-format off
/**
 * Id-free broadcast mul: C = A * broadcast(B). Shorthand for any_tiles_bcast<ELWMUL, tBcastDim>. See
 * any_tiles_bcast for the full parameter table and broadcast semantics. The most common broadcast use.
 *
 * | Param Type | Name           | Description                              | Type          | Valid Range | Required |
 * |------------|----------------|------------------------------------------|---------------|-------------|----------|
 * | Template   | tBcastDim      | The broadcast dim (ROW / COL / SCALAR)   | BroadcastType | N/A         | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode           | bool          | N/A         | False    |
 * | Function   | a / b          | Input operands                           | LLKOperand    | N/A         | True     |
 * | Function   | itile0 / itile1| Tile indices within A / B                | uint32_t      | N/A         | True     |
 * | Function   | idst           | DST register index for the result        | uint32_t      | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <
    BroadcastType tBcastDim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat AFormat,
    TensorShape AShape,
    DataFormat BFormat,
    TensorShape BShape>
ALWI void mul_tiles_bcast(
    LLKOperand<AFormat, AShape> a,
    LLKOperand<BFormat, BShape> b,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t bcast_row_idx = 0) {
    any_tiles_bcast<EltwiseBinaryType::ELWMUL, tBcastDim, is_fp32_dest_acc_en>(
        a, b, itile0, itile1, idst, bcast_row_idx);
}

}  // namespace experimental
}  // namespace ckernel
