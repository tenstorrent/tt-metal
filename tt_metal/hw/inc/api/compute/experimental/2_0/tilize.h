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
    UNPACK((llk_unpack_tilize_uninit<LLKOperand<InFormat, InShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en, PackMode::Default>()));
}

}  // namespace experimental
}  // namespace ckernel
