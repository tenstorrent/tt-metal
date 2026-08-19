// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_mem_descriptor.h"

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

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free tilize init. Takes an input and an output LLKOperand (data format + tile geometry
 * as NTTPs, deduced from the arguments) instead of CB ids; register formats are derived INSIDE the LLK. The
 * tilize pack init needs the input format (8-bit tilize workaround) plus the output format/geometry, hence
 * both operands. `block` is the tilize block width (ct_dim) the unpacker MOP is configured for.
 *
 * | Template | InFormat/InShape   | Input buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Template | OutFormat/OutShape | Output buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Function | block              | Tilize block width (ct_dim)                   | uint32_t | > 0 | True |
 */
// clang-format on
template <DataFormat InFormat, TensorShape InShape, DataFormat OutFormat, TensorShape OutShape>
ALWI void tilize_init(
    LLKOperand<InFormat, InShape> /*in*/, std::uint32_t block, LLKOperand<OutFormat, OutShape> /*out*/) {
    UNPACK((llk_unpack_tilize_init<LLKOperand<InFormat, InShape>::descriptor, DST_ACCUM_MODE>(block)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<InFormat, InShape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Tilize>()));
    PACK((llk_pack_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          LLKOperand<InFormat, InShape>::descriptor,
          DST_ACCUM_MODE,
          PackMode::Tilize>(1 /* num_tiles */)));
}

// clang-format off
/**
 * Experimental id-free tilize of one block. The op owns the block loop and self-syncs Dest per tile (no
 * kernel tile_regs). Runtime "where": in.l1_address (unpack base -- the LLK offsets by tile_index inside)
 * and out.l1_address (the block's first output tile).
 *
 * PER-TILE OUTPUT ADDRESSING + ASSUMPTION vs the legacy BH tilize (fifo_page_size == one tile_size):
 *   Tile t is packed to out.l1_address + t * <output tile stride>. The stride folds to a compile-time
 *   constant from the output descriptor via SCALE_DATUM_SIZE:
 *       stride_words = SCALE_DATUM_SIZE(OutFormat, OutShape.total_tensor_size()) >> 4   // bytes -> 16B words
 *   SCALE_DATUM_SIZE returns the tile size in BYTES (datum_count x the format's datum width: Float32 x4,
 *   Float16/Float16_b/UInt16 x2, else x1); L1 pack addresses are in 16-byte words, hence >> 4.
 *
 *   DISCREPANCY vs legacy: legacy BH tilize_block packs via llk_pack(0, ocb, t + output_tile_index), whose
 *   address advances by the CB's ACTUAL fifo_page_size (read from the CB interface). This 2.0 API has no CB
 *   handle, so it ASSUMES fifo_page_size == a single tile's size (the SCALE_DATUM_SIZE value above). That
 *   assumption holds when the output CB page is exactly one bare tile -- TRUE for linear formats
 *   (Float32 / Float16 / int, incl. the tested bf16) -- but NOT when the page differs from one bare tile:
 *     - block formats (Bfp8/Bfp4): the CB page includes shared-exponent bytes SCALE_DATUM_SIZE omits;
 *     - CB pages sized with alignment/padding, or holding more than one tile.
 *   In those cases the derived stride diverges from fifo_page_size and block>1 output would be mis-placed.
 *   KNOWN LIMITATION: a general fix must read the CB fifo_page_size (as legacy does) or add the exponent
 *   section to the stride. block==1 (the current test path) is unaffected because t is always 0.
 *
 * | Function | in    | Input operand (unpack base; LLK offsets by tile_index) | LLKOperand |     | True |
 * | Function | block | Number of column tiles in the block                    | uint32_t   | > 0 | True |
 * | Function | out   | Output operand (first tile's L1 address)               | LLKOperand |     | True |
 */
// clang-format on
template <DataFormat InFormat, TensorShape InShape, DataFormat OutFormat, TensorShape OutShape>
ALWI void tilize_block(LLKOperand<InFormat, InShape> in, std::uint32_t block, LLKOperand<OutFormat, OutShape> out) {
    // Unpack the whole block into srcA first (mirrors llk_unpack_tilize_block), then drain via math+pack.
    for (std::uint32_t t = 0; t < block; t++) {
        UNPACK((llk_unpack_tilize<LLKOperand<InFormat, InShape>::descriptor, DST_ACCUM_MODE>(in.l1_address, t)));
    }

    for (std::uint32_t t = 0; t < block; t++) {
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        MATH((llk_math_eltwise_unary_datacopy<
              LLKOperand<InFormat, InShape>::descriptor,
              DataCopyType::A2D,
              DST_ACCUM_MODE,
              BroadcastType::NONE,
              UnpackToDestEn>(0 /*dst index*/)));
        // Per-tile output slot: out.l1_address + t * (tile bytes / 16). The stride folds to a constant.
        PACK((llk_pack<
              LLKOperand<OutFormat, OutShape>::descriptor,
              DST_ACCUM_MODE,
              true /*out_of_order*/,
              PackMode::Default>(
            0 /*tile index*/,
            out.l1_address +
                t * (SCALE_DATUM_SIZE(static_cast<std::uint32_t>(OutFormat), OutShape.total_tensor_size()) >> 4))));

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Experimental id-free tilize uninit. Restore the SrcA/tile config so a subsequent op can reprogram the
 * unpacker, and reset the packer to Default mode.
 */
// clang-format on
template <DataFormat InFormat, TensorShape InShape, DataFormat OutFormat, TensorShape OutShape>
ALWI void tilize_uninit(LLKOperand<InFormat, InShape> /*in*/, LLKOperand<OutFormat, OutShape> /*out*/) {
    UNPACK((llk_unpack_tilize_uninit<LLKOperand<InFormat, InShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE, PackMode::Default>()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
