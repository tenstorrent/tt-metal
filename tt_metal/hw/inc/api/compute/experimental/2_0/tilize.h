// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/internal/op_hooks.h"
#include "api/compute/reg_api.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free tilize init, Quasar. Unpack TRISC programs an Unp0 buffer descriptor at in's base and the unpack-tilize
 * strides + MOP for a row-major block `block` tiles wide (tiles go to SrcA); math TRISC programs the SrcA -> Dest
 * datacopy; pack TRISC programs a one-tile pack to out (as pack_init). in's / out's bases are the DFB ring bases, so
 * one init serves every front() / back() of the rings. Re-init when another op ran in between (Quasar rule). No
 * uninit. 32x32 tiles; input Float16_b / Float16 (either Dest width), output as pack_init.
 *
 * | Param Type | Name                | Description                                       | Type        | Valid Range        | Required |
 * |------------|---------------------|---------------------------------------------------|-------------|--------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        | DST_ACCUM_MODE     | False    |
 * | Template   | InFormat / InShape  | Input (row-major) L1 format + tile geometry (deduced) | DataFormat/TensorShape | Float16_b, Float16; 32x32 | True |
 * | Template   | OutFormat / OutShape | Output L1 format + tile geometry (deduced)       | DataFormat/TensorShape | as pack_init; 32x32 | True |
 * | Function   | in                  | Input operand (row-major rows `block` tiles wide)  | LLKOperand  | N/A                | True     |
 * | Template   | block               | Block width in tiles (row-major row width)        | uint32_t    | > 0                | True     |
 * | Function   | out                 | Output operand (tiles)                            | LLKOperand  | N/A                | True     |
 */
// clang-format on
template <
    std::uint32_t block,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
inline void tilize_init(LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out) {
    static_assert(
        llk_2_0::is_supported_format(InFormat, false), "tilize_init (Quasar 2.0): input Float16_b / Float16 only.");
    static_assert(
        llk_2_0::is_supported_format(OutFormat, DST_ACCUM_MODE),
        "tilize_init (Quasar 2.0): output Float16_b / Float16, or Float32 at a 32-bit Dest.");
    static_assert(
        detail::is_32x32(InShape) && detail::is_32x32(OutShape), "tilize_init (Quasar 2.0): 32x32 tiles only.");
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "tilize_init: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    static_assert(block > 0, "tilize_init: block must be > 0");
    constexpr LLKMemDescriptor DIn = LLKOperand<InFormat, InShape>::descriptor;
    constexpr LLKMemDescriptor DOut = LLKOperand<OutFormat, OutShape>::descriptor;
    UNPACK((llk_2_0::unpack_tilize_init<DIn, is_fp32_dest_acc_en>(in.l1_address, block)));
    MATH((llk_2_0::math_datacopy_init<DIn, is_fp32_dest_acc_en>()));
    PACK((llk_2_0::pack_init<DOut, is_fp32_dest_acc_en, false /*relu*/>(out.l1_address)));
}

// clang-format off
/**
 * Id-free tilize of one block, Quasar: unpacks `block` tiles from the row-major input (tile input_tile_index of the
 * block-wide rows, i.e. row group input_tile_index / block, column input_tile_index % block, counted from in's first
 * tile), then per tile copies it to Dest and packs it to out tile output_tile_index + t. Self-syncs Dest per tile
 * (tile_regs_acquire .. release inside): call outside a kernel Dest section. Contiguous operands (stride 1): bind
 * their DFBs BLOCKED or 1:1, since a multi-thread STRIDED binding gives stride_tiles > 1.
 *
 * | Param Type | Name                | Description                                       | Type        | Valid Range        | Required |
 * |------------|---------------------|---------------------------------------------------|-------------|--------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        | DST_ACCUM_MODE     | False    |
 * | Function   | in                  | Input operand (its base = the tilize_init's)      | LLKOperand  | N/A                | True     |
 * | Template   | block               | Block width in tiles, as tilize_init              | uint32_t    | > 0                | True     |
 * | Function   | out                 | Output operand (its base = the tilize_init's)     | LLKOperand  | N/A                | True     |
 * | Function   | input_tile_index    | First input tile (multiple of block for whole rows) | uint32_t  | N/A                | False    |
 * | Function   | output_tile_index   | First output tile within out                      | uint32_t    | N/A                | False    |
 */
// clang-format on
template <
    std::uint32_t block,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
inline void tilize_block(
    LLKOperand<InFormat, InShape> in,
    LLKOperand<OutFormat, OutShape> out,
    std::uint32_t input_tile_index = 0,
    std::uint32_t output_tile_index = 0) {
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "tilize_block: Dest mode is kernel-wide (DST_ACCUM_MODE)");
    constexpr LLKMemDescriptor DIn = LLKOperand<InFormat, InShape>::descriptor;
    constexpr LLKMemDescriptor DOut = LLKOperand<OutFormat, OutShape>::descriptor;
    // L1 rows one tile wide (32 datums): a tile holds tile_rows of them; the block's row group is block * tile_rows.
    constexpr std::uint32_t tile_rows = InShape.num_faces_r_dim * InShape.face_r_dim;
    LLK_ASSERT(in.stride_tiles == 1 && out.stride_tiles == 1, "tilize_block (Quasar 2.0): contiguous operands only");
    UNPACK((llk_2_0::unpack_tilize<DIn>(
        in.l1_address,
        (in.tile(0) + (input_tile_index / block) * block) * tile_rows + input_tile_index % block,
        block)));
    for (std::uint32_t t = 0; t < block; ++t) {
        tile_regs_acquire();
        MATH((llk_2_0::math_datacopy(0)));
        tile_regs_commit();
        tile_regs_wait();
        PACK((llk_2_0::pack<DOut>(out.l1_address, 0, out.tile(output_tile_index + t))));
        tile_regs_release();
    }
}

}  // namespace experimental
}  // namespace ckernel

#else  // Blackhole

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
 * (template) is the tilize block width (ct_dim) the unpacker MOP is configured for. Blackhole only.
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                     | bool                   |  | False |
 * | Template | InFormat/InShape   | Input buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Template | OutFormat/OutShape | Output buffer L1 format + geometry (deduced)  | DataFormat/TensorShape |  | True |
 * | Template | block              | Tilize block width (ct_dim)                   | uint32_t | > 0 | True |
 */
// clang-format on
template <
    std::uint32_t block,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void tilize_init(LLKOperand<InFormat, InShape> /*in*/, LLKOperand<OutFormat, OutShape> /*out*/) {
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
 * | Template | block             | Number of column tiles in the block                             | uint32_t   | > 0  | True  |
 * | Function | out               | Output operand (block's first-tile L1 address)                  | LLKOperand |      | True  |
 * | Function | input_tile_index  | Starting tile index added to the unpack source index            | uint32_t   | >= 0 | False |
 * | Function | output_tile_index | Starting tile index added to the pack destination slot          | uint32_t   | >= 0 | False |
 */
// clang-format on
template <
    std::uint32_t block,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void tilize_block(
    LLKOperand<InFormat, InShape> in,
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

#endif  // ARCH_QUASAR
