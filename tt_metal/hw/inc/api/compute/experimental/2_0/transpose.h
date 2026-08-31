// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "data_format_derive.h"  // ckernel::infer_unpack_dst_format -- pure constexpr, used unconditionally below

#ifdef TRISC_MATH
#include "experimental/2_0/llk_math_unary_datacopy.h"
#include "llk_math_transpose_dest_api.h"
#endif

#ifdef TRISC_UNPACK
#include "experimental/2_0/llk_unpack_A.h"
#include "llk_unpack_common_api.h"  // llk_unpack_set_srcb_dummy_valid
#endif

// =====================================================================================================
// Id-free (2.0) transpose (input transpose / transpose WH). Takes a single input LLKOperand whose data
// format + tile geometry are NTTPs. Mirrors the legacy Blackhole transpose.h 3-way init split (unpack-to-dest
// for 32-bit dst formats, the int-FPU A2D reconstruct path for 8-bit integer formats, default otherwise) --
// but the split is resolved via `if constexpr` on the operand's compile-time Format instead of a runtime CB
// format lookup, since there is no CB id here. The underlying LLKs (id-free llk_unpack_A / id-free
// llk_math_eltwise_unary_datacopy, plus the format-free llk_math_transpose_dest reused as-is) are exactly
// the ones tile_move_copy.h's copy_tile already uses. Blackhole only; the Quasar transpose path is a
// completely separate LLK generation and is not ported here.
// =====================================================================================================

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Paired init function for transpose_tile / transpose_block. Takes an LLKOperand whose data format + tile
 * geometry are NTTPs (deduced from the argument). Selects, entirely at compile time, among the three
 * unpack/math configurations the legacy Blackhole transpose_init selects at runtime from the CB's formats:
 * unpack-to-dest for 32-bit dst formats (Float32/UInt32/Int32), the int-FPU (ELWADD) A2D reconstruct path
 * for 8-bit integer formats (Int8/UInt8), and the default path for everything else. compute_kernel_hw_startup
 * must already have run once. No CB id, no register format on the API surface.
 *
 * | Param Type | Name   | Description                                                   | Type        | Valid Range | Required |
 * |------------|--------|----------------------------------------------------------------|-------------|-------------|----------|
 * | Template   | Format | Buffer L1 data format (deduced from the LLKOperand argument)   | DataFormat  | N/A         | True     |
 * | Template   | Shape  | Tile geometry (deduced from the LLKOperand argument)           | TensorShape | N/A         | True     |
 * | Function   | src    | The source L1 operand (format + shape; address unused here)   | LLKOperand  | N/A         | True     |
 */
// clang-format on
namespace detail {
// True iff transpose takes the unpack-to-dest (A2D) path: the operand's REGISTER format (after the datacopy
// Float32-rebias via infer_unpack_dst_format) is 32-bit. Shared by transpose_init / transpose_tile. NOTE this
// runs Format through infer_unpack_dst_format first, unlike the unary-bcast select which tests Format directly.
template <DataFormat Format>
constexpr bool transpose_unpack_to_dest() {
    return is_32bit_format(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE));
}
}  // namespace detail

template <DataFormat Format, TensorShape Shape>
ALWI void transpose_init(LLKOperand<Format, Shape> /*src*/) {
    static_assert(
        is_legal_tile_shape(Shape),
        "transpose_init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");

    constexpr bool enable_unpack_to_dest = detail::transpose_unpack_to_dest<Format>();
    // Low-nibble compare intentionally matches both signed Int8 (14) and unsigned UInt8 (30 -> low nibble
    // 0xE): both 8-bit integer formats need the int-FPU (ELWADD) A2D reconstruct path.
    constexpr bool is_8bit_int =
        (static_cast<std::uint8_t>(Format) & 0xf) == static_cast<std::uint8_t>(DataFormat::Int8);

    if constexpr (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A_init<
                LLKOperand<Format, Shape>::descriptor,
                DST_ACCUM_MODE,
                BroadcastType::NONE,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE,
                UnpackToDestEn>(true /*transpose_of_faces*/, false /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE>()));
        MATH((llk_math_transpose_dest_init<false, true>()));
    } else {
        // Non-unpack-to-dest path (default + 8-bit integer). The unpack init is identical for both; the only
        // difference is that 8-bit integer (Int8/UInt8) transpose needs the int-FPU (ELWADD) A2D reconstruct
        // path, selected via the datacopy init's is_int_fpu_en NTTP (== is_8bit_int; false for the default
        // path). Ideally the LLK layer would infer this from the data format. TODO: #46832 (same limitation as
        // the legacy op).
        UNPACK((llk_unpack_A_init<
                LLKOperand<Format, Shape>::descriptor,
                DST_ACCUM_MODE,
                BroadcastType::NONE,
                true /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE>(true /*transpose_of_faces*/, true /*within_face_16x16_transpose*/)));
        MATH((llk_math_eltwise_unary_datacopy_init<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE,
              is_8bit_int /*is_int_fpu_en*/>()));
    }
}

// clang-format off
/**
 * Performs a 32x32 transpose operation *B[w,h] = A[h,w]* on one tile from the L1 region described by the
 * LLKOperand and writes the result to DST[idst]. DST must be in the acquired state (tile_regs_acquire).
 * This call is blocking and is only available on the compute engine. Pair with transpose_init.
 *
 * | Param Type | Name   | Description                                                  | Type        | Valid Range                                    | Required |
 * |------------|--------|---------------------------------------------------------------|-------------|-------------------------------------------------|----------|
 * | Template   | Format | Buffer L1 data format (deduced from the LLKOperand argument)  | DataFormat  | N/A                                             | True     |
 * | Template   | Shape  | Tile geometry (deduced from the LLKOperand argument)          | TensorShape | N/A                                             | True     |
 * | Function   | src    | The source L1 operand (format + shape + base address)         | LLKOperand  | N/A                                             | True     |
 * | Function   | itile  | Index of the tile A within `src`, relative to its base address| uint32_t    | N/A                                             | True     |
 * | Function   | idst   | Index of the tile in DST REG for the result B                 | uint32_t    | Must be less than the acquired size of DST REG | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void transpose_tile(LLKOperand<Format, Shape> src, std::uint32_t itile, std::uint32_t idst) {
    static_assert(
        is_legal_tile_shape(Shape),
        "transpose_tile: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");

    constexpr bool enable_unpack_to_dest = detail::transpose_unpack_to_dest<Format>();
    const std::uint32_t addr = detail::tile_address(src, itile);

    if constexpr (enable_unpack_to_dest) {
        UNPACK((llk_unpack_A<
                LLKOperand<Format, Shape>::descriptor,
                DST_ACCUM_MODE,
                BroadcastType::NONE,
                false /*acc_to_dest*/,
                EltwiseBinaryReuseDestType::NONE,
                UnpackToDestEn>(addr)));
        UNPACK((llk_unpack_set_srcb_dummy_valid()));
        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE,
              UnpackToDestEn>(idst)));
        MATH((llk_math_transpose_dest<false, true>(idst)));
    } else {
        UNPACK((llk_unpack_A<
                LLKOperand<Format, Shape>::descriptor,
                DST_ACCUM_MODE,
                BroadcastType::NONE,
                false /*acc_to_dest*/>(addr)));
        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<Format, Shape>::descriptor,
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE>(idst)));
    }
}

// clang-format off
/**
 * Performs a 32x32 transpose operation *B[w,h] = A[h,w]* on `ntiles` consecutive tiles from the L1 region
 * described by the LLKOperand, writing each result to a consecutive DST register slot. Block/loop form of
 * transpose_tile: tile `start_itile+i` lands in DST slot `start_idst+i`. Reuses the existing 2.0
 * transpose_tile per tile. Requires the same init as transpose_tile (transpose_init). DST must be in the
 * acquired state. This call is blocking and is only available on the compute engine.
 *
 * NOTE: The loop implementation is transitional, matching the legacy transpose_block. In the future this
 * for-loop must be folded into a hardware MOP / REPLAY buffer so the whole block issues as a single packed
 * op; the blocking then lives in llk-lib without changing this signature.
 *
 * | Param Type | Name        | Description                                                      | Type        | Valid Range                                    | Required |
 * |------------|-------------|-------------------------------------------------------------------|-------------|-------------------------------------------------|----------|
 * | Template   | Format      | Buffer L1 data format (deduced from the LLKOperand argument)      | DataFormat  | N/A                                             | True     |
 * | Template   | Shape       | Tile geometry (deduced from the LLKOperand argument)              | TensorShape | N/A                                             | True     |
 * | Function   | src         | The source L1 operand (format + shape + base address)             | LLKOperand  | N/A                                             | True     |
 * | Function   | start_itile | Index of the first tile A within `src`, relative to its base addr | uint32_t    | N/A                                             | True     |
 * | Function   | start_idst  | Index of the first destination tile in DST REG                    | uint32_t    | Must be less than the acquired size of DST REG | True     |
 * | Function   | ntiles      | The number of consecutive tiles to transpose                      | uint32_t    | start_idst + ntiles <= acquired DST REG size   | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void transpose_block(
    LLKOperand<Format, Shape> src, std::uint32_t start_itile, std::uint32_t start_idst, std::uint32_t ntiles) {
    static_assert(
        is_legal_tile_shape(Shape),
        "transpose_block: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        transpose_tile(src, start_itile + i, start_idst + i);
    }
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
