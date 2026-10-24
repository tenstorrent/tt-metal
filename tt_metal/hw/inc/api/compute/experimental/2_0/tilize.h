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
#include "experimental/2_0/llk_unpack_tilize.h"
#endif

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_tile.h"
#endif

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free tilize init. Takes an input and an output LLKOperand (data format + tile geometry as NTTPs,
 * deduced from the arguments) instead of CB ids; register formats are derived inside the LLK. Both operands
 * are required: the tilize pack init needs the input format as well as the output format/geometry. `block`
 * is the tilize block width (ct_dim) the unpacker MOP is configured for. Blackhole only.
 *
 * Compile-time rejected: an output geometry that differs from the input (tilize keeps the tile shape), input tiles
 * with 1 face or a single face column (narrow), (for a non-8-bit input) a tile with faces shorter than 16 rows
 * other than a two-face tile, a block-float input (block-float data has no row-major form), and a block-float
 * output shorter than 32 rows (unsupported on the BH compute datapath, as in pack / copy).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                     | bool                   |  | False |
 * | Template | InFormat/InShape   | Input buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Template | OutFormat/OutShape | Output buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Function | block              | Tilize block width (ct_dim)                   | uint32_t | > 0 | True |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void tilize_init(
    LLKOperand<InFormat, InShape> /*in*/, std::uint32_t block, LLKOperand<OutFormat, OutShape> /*out*/) {
    static_assert(is_legal_tile_shape(InShape), "tilize_init: illegal input tile shape.");
    static_assert(is_legal_tile_shape(OutShape), "tilize_init: illegal output tile shape.");
    static_assert(
        is_supported_tilize_operands(InFormat, InShape, OutShape),
        "tilize_init: unsupported tilize operands: the output geometry must equal the input geometry, the input "
        "must be two faces wide (2 or 4 faces, no narrow tile), and with a non-8-bit input a tile with faces "
        "shorter than 16 rows must have exactly two faces (the tilize unpack / pack programs cover no other "
        "operands).");
    static_assert(
        !is_block_float_format(InFormat),
        "tilize_init: unsupported input format: block-float data is tile-layout only (its shared exponent belongs to "
        "a 16-datum face row), so there is no row-major form for the tilize unpacker to read; tilize into a "
        "block-float output instead.");
    static_assert(
        !(is_block_float_format(OutFormat) && is_partial_height(OutShape)),
        "tilize_init: unsupported output: sub-32-row (partial-height) block-float tiles are not supported on the BH "
        "compute datapath; use a full 32-row tile.");
    UNPACK((llk_unpack_tilize_init<LLKOperand<InFormat, InShape>::descriptor, is_fp32_dest_acc_en>(block)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<InFormat, InShape>::descriptor,
          DataCopyType::A2D,
          is_fp32_dest_acc_en,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Tilize>()));
    PACK((llk_pack_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          LLKOperand<InFormat, InShape>::descriptor,
          is_fp32_dest_acc_en,
          PackMode::Tilize>(1 /* num_tiles */)));
}

// clang-format off
/**
 * Id-free tilize of one block. Owns the block loop and self-syncs DEST per tile (no kernel tile_regs). Reads
 * from in.l1_address (unpack base; the LLK offsets by tile index inside) and writes the block's tiles
 * starting at out.l1_address. Requires both an input and output operand. Blackhole only.
 *
 * Per-tile output stride is one tile's L1 size, derived at compile time from the output descriptor's format
 * and geometry -- this matches the shipping tilize factories' one-tile output CB page. input_tile_index and
 * output_tile_index (both default 0) shift the unpack source / pack destination tile index; at their
 * defaults, tile t lands in slot t and the source base is in.l1_address, byte-identical to the legacy op.
 *
 * Compile-time rejected: an output geometry that differs from the input (tilize keeps the tile shape), input tiles
 * with 1 face or a single face column (narrow), (for a non-8-bit input) a tile with faces shorter than 16 rows
 * other than a two-face tile, a block-float input (block-float data has no row-major form), and a block-float
 * output shorter than 32 rows (unsupported on the BH compute datapath, as in pack / copy).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                                       | bool       |      | False |
 * | Function | in                | Input operand (unpack base; LLK offsets by input_tile_index + t) | LLKOperand |      | True  |
 * | Function | block             | Number of column tiles in the block                             | uint32_t   | > 0  | True  |
 * | Function | out               | Output operand (block's first-tile L1 address)                  | LLKOperand |      | True  |
 * | Function | input_tile_index  | Starting tile index added to the unpack source index            | uint32_t   | >= 0 | False |
 * | Function | output_tile_index | Starting tile index added to the pack destination slot          | uint32_t   | >= 0 | False |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void tilize_block(
    LLKOperand<InFormat, InShape> in,
    std::uint32_t block,
    LLKOperand<OutFormat, OutShape> out,
    std::uint32_t input_tile_index = 0,
    std::uint32_t output_tile_index = 0) {
    static_assert(is_legal_tile_shape(InShape), "tilize_block: illegal input tile shape.");
    static_assert(is_legal_tile_shape(OutShape), "tilize_block: illegal output tile shape.");
    static_assert(
        is_supported_tilize_operands(InFormat, InShape, OutShape),
        "tilize_block: unsupported tilize operands: the output geometry must equal the input geometry, the input "
        "must be two faces wide (2 or 4 faces, no narrow tile), and with a non-8-bit input a tile with faces "
        "shorter than 16 rows must have exactly two faces (the tilize unpack / pack programs cover no other "
        "operands).");
    static_assert(
        !is_block_float_format(InFormat),
        "tilize_block: unsupported input format: block-float data is tile-layout only (its shared exponent belongs to "
        "a 16-datum face row), so there is no row-major form for the tilize unpacker to read; tilize into a "
        "block-float output instead.");
    static_assert(
        !(is_block_float_format(OutFormat) && is_partial_height(OutShape)),
        "tilize_block: unsupported output: sub-32-row (partial-height) block-float tiles are not supported on the BH "
        "compute datapath; use a full 32-row tile.");
    // Unpack the whole block into srcA first (mirrors llk_unpack_tilize_block), then drain via math+pack.
    for (std::uint32_t t = 0; t < block; t++) {
        UNPACK((llk_unpack_tilize<LLKOperand<InFormat, InShape>::descriptor, is_fp32_dest_acc_en>(
            in.l1_address, input_tile_index + t)));
    }

    for (std::uint32_t t = 0; t < block; t++) {
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<InFormat, InShape>::descriptor,
              DataCopyType::A2D,
              is_fp32_dest_acc_en,
              BroadcastType::NONE,
              UnpackToDestEn>(0 /*dst index*/)));
        // Per-tile output slot: out.l1_address + (output_tile_index + t) * one-tile L1 size (via the shared
        // tile_address helper; the stride folds to a compile-time constant).
        PACK((llk_pack<
              LLKOperand<OutFormat, OutShape>::descriptor,
              is_fp32_dest_acc_en,
              true /*out_of_order*/,
              PackMode::Default>(0 /*tile index*/, detail::tile_address(out, output_tile_index + t))));

        MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
        PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
    }
}

// clang-format off
/**
 * Id-free tilize uninit. Restores the SrcA/tile config so a subsequent op can reprogram the unpacker, and
 * resets the packer to Default mode. Blackhole only.
 *
 * Compile-time rejected: an output geometry that differs from the input (tilize keeps the tile shape), input tiles
 * with 1 face or a single face column (narrow), (for a non-8-bit input) a tile with faces shorter than 16 rows
 * other than a two-face tile, a block-float input (block-float data has no row-major form), and a block-float
 * output shorter than 32 rows (unsupported on the BH compute datapath, as in pack / copy).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode | bool | | False |
 */
// clang-format on
template <
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void tilize_uninit(LLKOperand<InFormat, InShape> /*in*/, LLKOperand<OutFormat, OutShape> /*out*/) {
    static_assert(
        is_supported_tilize_operands(InFormat, InShape, OutShape),
        "tilize_uninit: unsupported tilize operands: the output geometry must equal the input geometry, the input "
        "must be two faces wide (2 or 4 faces, no narrow tile), and with a non-8-bit input a tile with faces "
        "shorter than 16 rows must have exactly two faces (the tilize unpack / pack programs cover no other "
        "operands).");
    static_assert(
        !is_block_float_format(InFormat),
        "tilize_uninit: unsupported input format: block-float data is tile-layout only (its shared exponent belongs "
        "to a 16-datum face row), so there is no row-major form for the tilize unpacker to read; tilize into a "
        "block-float output instead.");
    static_assert(
        !(is_block_float_format(OutFormat) && is_partial_height(OutShape)),
        "tilize_uninit: unsupported output: sub-32-row (partial-height) block-float tiles are not supported on the BH "
        "compute datapath; use a full 32-row tile.");
    UNPACK((llk_unpack_tilize_uninit<LLKOperand<InFormat, InShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en, PackMode::Default>()));
}

}  // namespace experimental
}  // namespace ckernel
