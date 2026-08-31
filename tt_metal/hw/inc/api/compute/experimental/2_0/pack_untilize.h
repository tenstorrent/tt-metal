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

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_untilize.h"
#include "experimental/2_0/llk_pack_tile.h"
#endif

namespace ckernel {
namespace experimental {

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free pack-untilize init. Takes an input and an output LLKOperand (data format + tile
 * geometry as NTTPs) instead of CB ids. Initializes all three threads (UNPACK/MATH/PACK) to move tilized
 * tiles CB -> DEST (datacopy) and pack them out row-major. Register formats are derived INSIDE the LLK.
 *
 * Mirrors the legacy pack_untilize_init (3-thread) + pack_untilize_dest_init PACK sequence, with one
 * substitution: the legacy PACK path calls llk_pack_reconfig_data_format(ocb) to program the packer format
 * registers from CB metadata; the id-free path calls the LLKOperand overload of that same op (formats from
 * OUT). llk_pack_untilize_init itself only programs addrmod/MOP/z-stride, so this format program is
 * required. BH DEST remap is (re)configured here (llk_math_reconfig_remap), matching the default legacy init.
 *
 * | Template | block_ct_dim/full_ct_dim | Block width / full input width in tiles | uint32_t | | False |
 * | Function | in                       | Input operand (tilized; format+geometry) | LLKOperand | | True |
 * | Function | out                      | Output operand (row-major destination)   | LLKOperand | | True |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_init(LLKOperand<InFormat, InShape> /*in*/, LLKOperand<OutFormat, OutShape> /*out*/) {
    static_assert(is_legal_tile_shape(InShape), "pack_untilize_init: illegal input tile shape.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_init: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_init: full_ct_dim must be a positive multiple of block_ct_dim.");
    // UNPACK + MATH: configure the CB -> DEST datacopy (input format drives the SrcA/Dest register format).
    UNPACK((llk_unpack_A_init<
            LLKOperand<InFormat, InShape>::descriptor,
            DST_ACCUM_MODE,
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            UnpackToDestEn>(0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<InFormat, InShape>::descriptor,
          DataCopyType::A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Default>()));

    // PACK: (re)configure BH DEST remap, program the packer output formats, then the untilize MOP/strides
    // and the untilize dest-offset registers.
    MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          DST_ACCUM_MODE,
          block_ct_dim,
          full_ct_dim>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
}

// clang-format off
/**
 * Experimental id-free pack-untilize of one block. The op owns the row/column loops and self-syncs DEST.
 * Runtime "where": in.l1_address (unpack base; the op offsets by the per-tile input stride derived from
 * InShape) and out.l1_address (the block's first row-major output tile).
 *
 * ADDRESSING vs the legacy BH pack_untilize (fifo_page_size == one tile_size):
 *   Legacy pack_untilize advances the packer write address across tile-rows by the CB's ACTUAL
 *   fifo_page_size (full_ct_dim * fifo_page_size, read from the CB interface). This id-free op has no CB
 *   handle, so llk_pack_untilize derives that per-row stride from the OUTPUT descriptor via
 *   tile_stride_words == one tile's L1 size; the input per-tile unpack base uses the same for InShape.
 *   The shipping untilize factories set both CB pages to one tile (in/out_single_tile_size), so this matches
 *   fifo_page_size for every format they use: geometry-exact for linear formats, exp section included for
 *   block floats. Remaining edge (no shipping op hits it): padded/multi-tile pages, and partial/tiny BFP
 *   tiles (tile_stride_words uses full-tile BFP size).
 *
 * | Template | block_ct_dim/full_ct_dim | Block width / full input width in tiles | uint32_t | | False |
 * | Function | in            | Input operand (tilized; unpack base)          | LLKOperand | | True |
 * | Function | block_rt_dim  | Height of the block in tiles (rows to pack)   | uint32_t   | >= 1 | True |
 * | Function | out           | Output operand (first row-major tile address) | LLKOperand | | True |
 * | Function | block_c_index | Block column index (when full_ct_dim > block_ct_dim) | uint32_t | >= 0 | False |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_block(
    LLKOperand<InFormat, InShape> in,
    std::uint32_t block_rt_dim,
    LLKOperand<OutFormat, OutShape> out,
    std::uint32_t block_c_index = 0) {
    static_assert(is_legal_tile_shape(InShape), "pack_untilize_block: illegal input tile shape.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_block: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_block: full_ct_dim must be a positive multiple of block_ct_dim.");
    // Per-tile input stride (16B words) == one tile's L1 size, folded to a compile-time constant via
    // tile_stride_words: geometry-exact for linear formats, exp section included for block floats.
    constexpr std::uint32_t in_tile_stride = tile_stride_words(InFormat, InShape);

    for (std::uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (std::uint32_t c = 0; c < block_ct_dim; ++c) {
            UNPACK((llk_unpack_A<
                    LLKOperand<InFormat, InShape>::descriptor,
                    DST_ACCUM_MODE,
                    BroadcastType::NONE,
                    false /*acc_to_dest*/,
                    EltwiseBinaryReuseDestType::NONE,
                    UnpackToDestEn>(in.l1_address + c * in_tile_stride)));
            MATH((llk_math_eltwise_unary_datacopy<
                  LLKOperand<InFormat, InShape>::descriptor,
                  DataCopyType::A2D,
                  DST_ACCUM_MODE,
                  BroadcastType::NONE,
                  UnpackToDestEn>(c)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE, block_ct_dim, full_ct_dim>(
            1 /*block_rt_dim*/, out.l1_address, block_c_index)));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Experimental id-free pack-untilize DEST init (PACK thread only). Use this when the tiles to untilize are
 * ALREADY resident in the DEST register (placed by copy_tile/reduce_tile/etc.), so UNPACK and MATH are NOT
 * configured for a CB -> DEST datacopy here (contrast pack_untilize_init, which configures all three threads).
 * Takes only the OUTPUT LLKOperand (data format + tile geometry as NTTPs) instead of a CB id; the packer
 * register format is derived INSIDE the LLK from OUT. Mirrors the legacy pack_untilize_dest_init PACK path:
 * (re)configure BH DEST remap, program the packer output format, program the untilize MOP/strides, then the
 * untilize dest-offset registers. Pair with pack_untilize_dest / pack_untilize_uninit.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range               | Required              |
 * |------------|---------------|---------------------------------------------------|------------|---------------------------|-----------------------|
 * | Template   | block_ct_dim  | Width of a single block in tiles                  | uint32_t   | 1 to max (DEST size)      | False (default = 8)   |
 * | Template   | full_ct_dim   | Width of a full input in tiles                    | uint32_t   | Divisible by block_ct_dim | False                 |
 * | Template   | diagonal      | Diagonal packing (unused on Blackhole; must be false) | bool   | false                     | False                 |
 * | Template   | narrow_row    | Whether the provided input is narrow              | bool       | true/false                | False                 |
 * | Template   | row_num_datums| Number of datums per row                          | uint32_t   | >= 1                      | False                 |
 * | Template   | dense         | Packs two 2-face tiles in a single 4-face region  | bool       | true/false                | False (default false) |
 * | Function   | out           | Output operand (row-major destination; format+geometry) | LLKOperand |                     | True                  |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_dest_init(LLKOperand<OutFormat, OutShape> /*out*/) {
    static_assert(diagonal == false, "pack_untilize_dest_init: diagonal is only supported on WH.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_dest_init: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_dest_init: full_ct_dim must be a positive multiple of block_ct_dim.");
    MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          DST_ACCUM_MODE,
          block_ct_dim,
          full_ct_dim,
          narrow_row,
          row_num_datums,
          dense>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
}

// clang-format off
/**
 * Experimental id-free pack-untilize of a block whose source is the DEST register (not a CB). Packs
 * (untilizes) block_rt_dim tile-rows that are ALREADY in DEST out to the row-major output at out.l1_address.
 * There is NO input LLKOperand: the source is DEST (selected by tile_dst_ct_offset / tile_dst_rt_offset).
 * The caller owns DEST synchronization (tile_regs_acquire/commit/wait/release) and any CB flow control. The
 * per-tile-row output stride is derived from the OUTPUT descriptor (tile_stride_words == one tile's L1 size;
 * see the addressing note on pack_untilize_block). Pair with pack_untilize_dest_init.
 *
 * | Param Type | Name               | Description                                                    | Type       | Valid Range                             | Required              |
 * |------------|--------------------|----------------------------------------------------------------|------------|-----------------------------------------|-----------------------|
 * | Template   | block_ct_dim       | Width of a single block in tiles                               | uint32_t   | 1 to max (DEST size)                    | False (default = 8)   |
 * | Template   | full_ct_dim        | Width of a full input in tiles                                 | uint32_t   | Divisible by block_ct_dim               | False                 |
 * | Template   | diagonal           | Diagonal packing (unused on Blackhole; must be false)          | bool       | false                                   | False                 |
 * | Template   | narrow_row         | Whether the provided input is narrow                           | bool       | true/false                              | False                 |
 * | Template   | row_num_datums     | Number of datums per row                                        | uint32_t   | >= 1                                    | False                 |
 * | Template   | tile_dst_ct_offset | Compile-time offset of the tile index in DEST from which to pack | uint32_t | 0 to 7 (0 to 3 if fp32 dest enabled)    | False (default = 0)   |
 * | Template   | dense              | Packs two 2-face tiles in a single 4-face region               | bool       | true/false                              | False (default false) |
 * | Function   | out                | Output operand (first row-major output tile address)          | LLKOperand |                                         | True                  |
 * | Function   | block_rt_dim       | Height of the block in tiles (rows to pack)                    | uint32_t   | >= 1                                    | False (default = 1)   |
 * | Function   | block_c_index      | Block column index (used when full_ct_dim > block_ct_dim)      | uint32_t   | >= 0                                    | False (default = 0)   |
 * | Function   | tile_dst_rt_offset | Runtime offset of the tile index in DEST from which to pack    | uint32_t   | 0 to 7 (0 to 3 if fp32 dest enabled)    | False (default = 0)   |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_dest(
    LLKOperand<OutFormat, OutShape> out,
    std::uint32_t block_rt_dim = 1,
    std::uint32_t block_c_index = 0,
    std::uint32_t tile_dst_rt_offset = 0) {
    static_assert(diagonal == false, "pack_untilize_dest: diagonal is only supported on WH.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_dest: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_dest: full_ct_dim must be a positive multiple of block_ct_dim.");
    PACK((llk_pack_untilize<
          LLKOperand<OutFormat, OutShape>::descriptor,
          DST_ACCUM_MODE,
          block_ct_dim,
          full_ct_dim,
          narrow_row,
          row_num_datums,
          tile_dst_ct_offset,
          dense>(block_rt_dim, out.l1_address, block_c_index, tile_dst_rt_offset)));
}

// clang-format off
/**
 * Experimental id-free pack-untilize uninit. Restores the packer Z stride (via the output descriptor's
 * register format) and resets the packer to Default mode so a subsequent op can reprogram it. Mirrors the
 * legacy pack_untilize_uninit (BH path).
 */
// clang-format on
template <DataFormat OutFormat, TensorShape OutShape>
ALWI void pack_untilize_uninit(LLKOperand<OutFormat, OutShape> /*out*/) {
    PACK((llk_pack_untilize_uninit<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, DST_ACCUM_MODE, PackMode::Default>()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
