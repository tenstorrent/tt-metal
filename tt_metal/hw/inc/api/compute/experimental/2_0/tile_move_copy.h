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

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free datacopy init. Takes an LLKOperand whose data format + tile geometry are NTTPs
 * (deduced from the argument). The op forwards LLKOperand<F,S>::descriptor to the LLK as an NTTP (folds);
 * the register format is derived INSIDE the LLK. It reprograms the unpacker + math datacopy for the operand's
 * format WITHOUT touching the packer dst-offset registers (those come from compute_kernel_hw_startup) -- on
 * Blackhole this matches the legacy copy_init unpacker + math datacopy re-init. No CB id, no register format on
 * the API surface. Legacy ckernel::copy_tile_init is untouched.
 *
 * fp8 (Fp8_e4m3 / Lf8 e5m2) IS supported: copy_init programs the format-driven Src zero-substitution flag
 * exactly as legacy copy_init does (ckernel::math::_configure_copy_zero_flag_state_) -- it flushes fp8 zeros
 * (whose SrcA residual would otherwise read back nonzero) and preserves the residual for all other formats
 * (e.g. bf16 -0.0 / 16b-int). The flag is derived id-free from the register format (infer_unpack_dst_format).
 *
 * PARITY: transpose / transpose_within_16x16_face restore the copy-with-transpose capability legacy copy_init
 * exposes; they were dropped when this id-free path was first authored (the authoring agent only exercised the
 * plain, non-transpose copy). Both default OFF (0), so the default copy path is byte-identical; when nonzero they
 * drive the unpacker exactly as legacy copy_init does (transpose -> transpose_of_faces, transpose_within_16x16_face
 * -> within_face_16x16_transpose on the id-free llk_unpack_A_init, the same args transpose_init uses).
 *
 * | Template | Format                      | Buffer L1 data format (deduced from the LLKOperand argument) | DataFormat  |                                                                     | True  |
 * | Template | Shape                       | Tile geometry (deduced from the LLKOperand argument)         | TensorShape |                                                                     | True  |
 * | Function | transpose                   | Flag to perform transpose on SrcA                           | uint32_t    | Any positive value will indicate transpose is set                   | False |
 * | Function | transpose_within_16x16_face | Flag to perform transpose within 16x16 face                 | uint32_t    | Any positive value will indicate transpose within 16x16 face is set | False |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_init(
    LLKOperand<Format, Shape> /*src*/, std::uint32_t transpose = 0, std::uint32_t transpose_within_16x16_face = 0) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_init: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    UNPACK((llk_unpack_A_init<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(transpose, transpose_within_16x16_face)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE>()));
    // Src zero-substitution flag, chosen by the derived register format (id-free equivalent of legacy
    // copy_init's ckernel::math::_configure_copy_zero_flag_state_(get_operand_dst_format(cbid))): preserve
    // bf16 -0.0 / 16b-int residuals, but flush fp8 (e4m3/e5m2) whose zero widens to a nonzero SrcA residual
    // and must read back as 0. MATH-only config; records no format-reconfig diff. The fn masks to 0x1F.
    MATH((ckernel::math::_configure_copy_zero_flag_state_(
        static_cast<std::uint32_t>(ckernel::infer_unpack_dst_format(Format, DST_ACCUM_MODE)))));
}

// clang-format off
/**
 * Experimental id-free datacopy. Copies one tile from the L1 region described by the LLKOperand into DST.
 * Compile-time "what" = the LLKOperand NTTPs (Format + Shape, fold/DCE); runtime "where" = src.l1_address
 * (from the address seam). No CB id, no register format on the API. Precondition: the DST register must be
 * acquired (tile_regs_acquire) before the call, and copy_init must have run.
 *
 * | Template | Format         | Buffer L1 data format (deduced from LLKOperand)   | DataFormat  |         | True |
 * | Template | Shape          | Tile geometry (deduced from LLKOperand)           | TensorShape |         | True |
 * | Function | src            | The source L1 operand (format+shape+address)      | LLKOperand  |         | True |
 * | Function | dst_tile_index | Tile index in the DST register                    | uint32_t    | 0 to 15 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_tile(LLKOperand<Format, Shape> src, std::uint32_t dst_tile_index) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_tile: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    UNPACK((llk_unpack_A<
            LLKOperand<Format, Shape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(src.l1_address)));
    MATH((llk_math_eltwise_unary_datacopy<
          LLKOperand<Format, Shape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          UnpackToDestEn>(dst_tile_index)));
}

// clang-format off
/**
 * Experimental id-free block datacopy. Copies `ntiles` consecutive tiles from the L1 region described by the
 * LLKOperand into consecutive DST register slots. Block/loop form of copy_tile: matches the legacy
 * copy_block semantics (tile start_in_tile_index+i lands in DST slot start_dst_tile_index+i). Reuses the
 * existing 2.0 copy_tile per tile; each tile's source address is the operand base offset by
 * (start_in_tile_index + i) * tile_stride_words(Format, Shape) 16-byte words (folds to a compile-time
 * constant). No CB id, no register format on the API. Requires the same init as copy_tile (copy_init).
 *
 * | Template | Format               | Buffer L1 data format (deduced from LLKOperand)             | DataFormat  |         | True |
 * | Template | Shape                | Tile geometry (deduced from LLKOperand)                    | TensorShape |         | True |
 * | Function | src                  | The source L1 operand (format+shape+base address)          | LLKOperand  |         | True |
 * | Function | start_in_tile_index  | Index of the first source tile, relative to src base        | uint32_t    | N/A     | True |
 * | Function | start_dst_tile_index | Index of the first destination tile in the DST register     | uint32_t    | 0 to 15 | True |
 * | Function | ntiles               | Number of consecutive tiles to copy                         | uint32_t    | start_dst_tile_index + ntiles <= 16 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void copy_block(
    LLKOperand<Format, Shape> src,
    std::uint32_t start_in_tile_index,
    std::uint32_t start_dst_tile_index,
    std::uint32_t ntiles) {
    static_assert(
        is_legal_tile_shape(Shape),
        "copy_block: illegal tile shape (face_r_dim must be 1/2/4/8/16, total faces 1/2/4).");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "copy: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath "
        "(they produce garbage even at tile 0); use a full 32-row tile.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        copy_tile(
            LLKOperand<Format, Shape>(detail::tile_address(src, start_in_tile_index + i)), start_dst_tile_index + i);
    }
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
