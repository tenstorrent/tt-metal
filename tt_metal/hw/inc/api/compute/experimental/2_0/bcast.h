// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#endif

// =====================================================================================================
// Id-free (2.0) unary broadcast (unary_bcast). Single-input op: unpacks one L1 tile (with the requested
// broadcast mode) and datacopies it into DST. Mirrors the legacy ckernel::unary_bcast<bcast_type> family
// (init/op/uninit) but takes an LLKOperand<Format,Shape> instead of a CB id: Format + Shape are compile-time
// NTTPs forwarded to the LLK as LLKOperand::descriptor (folds/DCE), and l1_address is the only runtime state.
// The register format is derived INSIDE the reused LLK overloads; no CB id / register format on the API.
//
// Broadcast mode / DataCopyType selection matches the legacy op exactly:
//   * BroadcastType::NONE is a pass-through and uses DataCopyType::A2D (the tile stays in SrcA; reading it
//     back with B2D would copy zeros and hang the unpacker).
//   * ROW / COL / SCALAR broadcasts leave the tile in SrcB and use DataCopyType::B2D.
//   * 32-bit formats (Float32/UInt32/Int32) take the unpack-to-dest A2D path (SrcB is only 19 bits wide);
//     this is derived at COMPILE time from the LLKOperand's Format (legacy derives it at runtime from the CB).
// Blackhole only. The Quasar-specific paths and the deprecated CB-id (icb, ocb) full inits are not carried over.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Paired init for experimental::unary_bcast. Configures the unpack + math pipeline for the id-free unary
 * broadcast op; call before unary_bcast (including when switching to it from another op). The one-time
 * hardware configuration must already have been performed via compute_kernel_hw_startup at kernel start.
 * The broadcast mode and A2D/B2D data-copy direction are fixed here from bcast_type + the operand Format.
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
    // 32-bit formats use the unpack-to-dest A2D path (SrcB is only 19 bits wide); folds to a constant.
    constexpr bool enable_unpack_to_dest =
        (Format == DataFormat::Float32) || (Format == DataFormat::UInt32) || (Format == DataFormat::Int32);
    constexpr DataCopyType dcopy =
        (enable_unpack_to_dest || bcast_type == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;
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
    constexpr bool enable_unpack_to_dest =
        (Format == DataFormat::Float32) || (Format == DataFormat::UInt32) || (Format == DataFormat::Int32);
    constexpr DataCopyType dcopy =
        (enable_unpack_to_dest || bcast_type == BroadcastType::NONE) ? DataCopyType::A2D : DataCopyType::B2D;
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
 * Paired uninit for experimental::unary_bcast. Restores the unpack + math pipeline so a following op can be
 * initialized cleanly. The underlying LLK uninits are format-free; the operand is taken only so the 32-bit
 * unpack-to-dest math-uninit variant is selected consistently with unary_bcast_init (folds to a constant).
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
    constexpr bool enable_unpack_to_dest =
        (Format == DataFormat::Float32) || (Format == DataFormat::UInt32) || (Format == DataFormat::Int32);
    UNPACK((llk_unpack_A_uninit<bcast_type>()));
    MATH((llk_math_eltwise_unary_datacopy_uninit<bcast_type, enable_unpack_to_dest>()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
