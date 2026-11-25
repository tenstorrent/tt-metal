// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/internal/op_hooks.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free pack-untilize Dest init, Quasar (pack TRISC only). Programs a Pack0 buffer descriptor at out's base, the
 * packer input format (Float32 with a 32-bit Dest), L1 accumulation off, and the untilize strides + MOP for
 * block_ct_dim Dest tiles into row-major rows full_ct_dim tiles wide. The tiles to untilize are put in Dest by an
 * earlier op (e.g. copy_block). out's base is the DFB ring base, so one init serves every back() of the ring.
 * Re-init when another op ran in between (Quasar rule). No uninit. 32x32 tiles; formats as pack_init.
 *
 * | Param Type | Name                | Description                                       | Type        | Valid Range               | Required |
 * |------------|---------------------|---------------------------------------------------|-------------|---------------------------|----------|
 * | Template   | block_ct_dim        | Width of a block in tiles                         | uint32_t    | 1 to Dest section tiles   | False (8) |
 * | Template   | full_ct_dim         | Width of a full row-major row in tiles            | uint32_t    | Multiple of block_ct_dim  | False (block_ct_dim) |
 * | Template   | diagonal            | Diagonal packing (WH only)                        | bool        | false                     | False    |
 * | Template   | narrow_row          | Narrow rows (not on Quasar)                       | bool        | false                     | False    |
 * | Template   | row_num_datums      | Datums per row                                    | uint32_t    | TILE_C_DIM                | False    |
 * | Template   | dense               | Two 2-face tiles per 4-face region (not on Quasar) | bool       | false                     | False    |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        | DST_ACCUM_MODE            | False    |
 * | Template   | OutFormat / OutShape | Output L1 format + tile geometry (deduced)       | DataFormat/TensorShape | as pack_init; 32x32 | True |
 * | Function   | out                 | Output operand (row-major destination)            | LLKOperand  | N/A                       | True     |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat OutFormat,
    TensorShape OutShape>
inline void pack_untilize_dest_init(LLKOperand<OutFormat, OutShape> out) {
    static_assert(
        !diagonal && !narrow_row && !dense && row_num_datums == TILE_C_DIM,
        "pack_untilize_dest_init (Quasar 2.0): diagonal / narrow_row / dense are not supported.");
    static_assert(
        llk_2_0::is_supported_format(OutFormat, DST_ACCUM_MODE),
        "pack_untilize_dest_init (Quasar 2.0): only Float16_b / Float16, or Float32 at a 32-bit Dest.");
    static_assert(detail::is_32x32(OutShape), "pack_untilize_dest_init (Quasar 2.0): 32x32 tiles only.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_dest_init: full_ct_dim must be a positive multiple of block_ct_dim.");
    static_assert(
        block_ct_dim <= detail::dest_section_tiles<is_fp32_dest_acc_en>(),
        "pack_untilize_dest_init: block_ct_dim exceeds the Dest section.");
    static_assert(is_fp32_dest_acc_en == DST_ACCUM_MODE, "pack_untilize_dest_init: Dest mode is kernel-wide");
    PACK((llk_2_0::pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          is_fp32_dest_acc_en,
          block_ct_dim,
          full_ct_dim>(out.l1_address)));
}

// clang-format off
/**
 * Id-free pack-untilize of a block from Dest, Quasar (pack TRISC only): for each of block_rt_dim tile rows r, packs
 * Dest tiles r * block_ct_dim + tile_dst_rt_offset + tile_dst_ct_offset .. + block_ct_dim - 1 into row-major columns
 * block_c_index * block_ct_dim .. of tile row r of out (rows full_ct_dim tiles wide, from out's first tile). Call
 * between tile_regs_wait and tile_regs_release; the caller owns Dest sync. Contiguous out (stride 1).
 *
 * | Param Type | Name                | Description                                       | Type        | Valid Range               | Required |
 * |------------|---------------------|---------------------------------------------------|-------------|---------------------------|----------|
 * | Template   | block_ct_dim        | As pack_untilize_dest_init                        | uint32_t    | as the init               | False (8) |
 * | Template   | full_ct_dim         | As pack_untilize_dest_init                        | uint32_t    | as the init               | False    |
 * | Template   | diagonal / narrow_row / row_num_datums / dense | As pack_untilize_dest_init | -       | as the init               | False    |
 * | Template   | tile_dst_ct_offset  | Compile-time Dest tile offset                     | uint32_t    | in the Dest section       | False (0) |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        | DST_ACCUM_MODE            | False    |
 * | Function   | out                 | Output operand (its base = the init's)            | LLKOperand  | N/A                       | True     |
 * | Template   | block_rt_dim        | Tile rows to pack                                 | uint32_t    | >= 1                      | False (1) |
 * | Function   | block_c_index       | Block column index within the full row            | uint32_t    | < full_ct_dim / block_ct_dim | False (0) |
 * | Function   | tile_dst_rt_offset  | Runtime Dest tile offset                          | uint32_t    | in the Dest section       | False (0) |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    std::uint32_t block_rt_dim = 1,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat OutFormat,
    TensorShape OutShape>
inline void pack_untilize_dest_block(
    LLKOperand<OutFormat, OutShape> out, std::uint32_t block_c_index = 0, std::uint32_t tile_dst_rt_offset = 0) {
    // L1 rows one tile wide (32 datums): a tile holds tile_rows of them; a row-major tile row is full_ct_dim wide.
    constexpr std::uint32_t tile_rows = OutShape.num_faces_r_dim * OutShape.face_r_dim;
    LLK_ASSERT(out.stride_tiles == 1, "pack_untilize_dest_block (Quasar 2.0): contiguous out only");
    LLK_ASSERT(
        block_rt_dim * block_ct_dim + tile_dst_rt_offset + tile_dst_ct_offset <=
            detail::dest_section_tiles<is_fp32_dest_acc_en>(),
        "pack_untilize_dest_block: block exceeds the Dest section");
    for (std::uint32_t r = 0; r < block_rt_dim; ++r) {
        PACK((llk_2_0::pack_untilize<LLKOperand<OutFormat, OutShape>::descriptor>(
            out.l1_address,
            r * block_ct_dim + tile_dst_rt_offset + tile_dst_ct_offset,
            (out.tile(0) + r * full_ct_dim) * tile_rows + block_c_index * block_ct_dim)));
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
#include "experimental/2_0/llk_unpack_A.h"
#endif

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_untilize.h"
#include "experimental/2_0/llk_pack_tile.h"
#endif

namespace ckernel {
namespace experimental {

namespace detail {

// The caller picks the path, as copy_src_to_dest_* / copy_direct_to_dest_*: Src (unpack to SrcA, then an A2D datacopy
// into Dest) for any register format that is not 32-bit, direct (unpack straight into Dest) for a 32-bit one.
template <
    bool direct,
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool is_fp32_dest_acc_en,
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
    static_assert(
        direct || !is_unpack_to_dest<InFormat, is_fp32_dest_acc_en>(),
        "pack_untilize_src_to_dest: a 32-bit register format (Float32 at fp32 Dest, Int32, UInt32) does not fit the "
        "19-bit SrcA; use pack_untilize_direct_to_dest_*.");
    static_assert(
        !direct || is_unpack_to_dest<InFormat, is_fp32_dest_acc_en>(),
        "pack_untilize_direct_to_dest: unpack-to-Dest takes only a 32-bit register format (Float32 at fp32 Dest, "
        "Int32, UInt32); use pack_untilize_src_to_dest_*.");
    // UNPACK + MATH: configure the CB -> DEST datacopy (input format drives the SrcA/Dest register format).
    UNPACK((llk_unpack_A_init<
            LLKOperand<InFormat, InShape>::descriptor,
            is_fp32_dest_acc_en,
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            direct>(0 /*transpose_of_faces*/, 0 /*within_face_16x16_transpose*/)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          LLKOperand<InFormat, InShape>::descriptor,
          DataCopyType::A2D,
          is_fp32_dest_acc_en,
          BroadcastType::NONE,
          false /*is_int_en*/,
          PackMode::Default>()));

    // PACK: (re)configure BH DEST remap, program the packer output formats, then the untilize MOP/strides
    // and the untilize dest-offset registers.
    MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((llk_pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          is_fp32_dest_acc_en,
          block_ct_dim,
          full_ct_dim>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
}

template <
    bool direct,
    std::uint32_t block_ct_dim,
    std::uint32_t full_ct_dim,
    bool is_fp32_dest_acc_en,
    std::uint32_t block_rt_dim,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_block(
    LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out, std::uint32_t block_c_index = 0) {
    static_assert(is_legal_tile_shape(InShape), "pack_untilize_block: illegal input tile shape.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_block: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_block: full_ct_dim must be a positive multiple of block_ct_dim.");
    static_assert(
        block_rt_dim == 1,
        "pack_untilize_block: block_rt_dim > 1 is not supported (no in/out row stride; would re-read "
        "and overwrite the first row).");
    // Per-tile input stride (16B words) == one tile's L1 size, folded to a compile-time constant via
    // tile_stride_words: geometry-exact for linear formats, exp section included for block floats.
    constexpr std::uint32_t in_tile_stride = tile_stride_words(InFormat, InShape);

    for (std::uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (std::uint32_t c = 0; c < block_ct_dim; ++c) {
            UNPACK((llk_unpack_A<
                    LLKOperand<InFormat, InShape>::descriptor,
                    is_fp32_dest_acc_en,
                    BroadcastType::NONE,
                    false /*acc_to_dest*/,
                    EltwiseBinaryReuseDestType::NONE,
                    direct>(in.l1_address + c * in_tile_stride)));
            MATH((llk_math_eltwise_unary_datacopy<
                  LLKOperand<InFormat, InShape>::descriptor,
                  DataCopyType::A2D,
                  is_fp32_dest_acc_en,
                  BroadcastType::NONE,
                  direct>(c)));
        }
        MATH((llk_math_dest_section_done<is_fp32_dest_acc_en>()));
        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<
              LLKOperand<OutFormat, OutShape>::descriptor,
              is_fp32_dest_acc_en,
              block_ct_dim,
              full_ct_dim>(1 /*block_rt_dim*/, out.l1_address, block_c_index)));
        PACK((llk_pack_dest_section_done<is_fp32_dest_acc_en>()));
    }
}

}  // namespace detail

// clang-format off
/**
 * Id-free pack-untilize init, Src path (unpack to SrcA, A2D datacopy). Takes an input and an output LLKOperand (data format + tile geometry as
 * NTTPs) instead of CB ids. Initializes all three threads (UNPACK/MATH/PACK) to move tilized tiles from the input
 * into DEST and pack them out row-major; register formats are derived inside the LLK. DEST remap is (re)configured
 * here. Input register format: not 32-bit (compile-time checked). Blackhole only.
 *
 * | Param Type | Name                     | Description                               | Type       | Valid Range | Required |
 * |------------|--------------------------|-------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim/full_ct_dim | Block width / full input width in tiles   | uint32_t   |             | False    |
 * | Template   | is_fp32_dest_acc_en      | Whether DEST is in fp32 accumulation mode | bool       |             | False    |
 * | Function   | in                       | Input operand (tilized; format+geometry)  | LLKOperand |             | True     |
 * | Function   | out                      | Output operand (row-major destination)    | LLKOperand |             | True     |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_src_to_dest_init(LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out) {
    detail::pack_untilize_init<false, block_ct_dim, full_ct_dim, is_fp32_dest_acc_en>(in, out);
}

// clang-format off
/**
 * Id-free pack-untilize of one block. Pair with pack_untilize_src_to_dest_init. Owns the column loop and self-syncs DEST. Reads from
 * in.l1_address (unpack base; offset per column tile by InShape's stride) and writes the block's first row-major
 * output tile to out.l1_address. block_rt_dim must be 1 (no in/out row stride). Blackhole only.
 *
 * | Param Type | Name                     | Description                                          | Type       | Valid Range | Required |
 * |------------|--------------------------|------------------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim/full_ct_dim | Block width / full input width in tiles              | uint32_t   |             | False    |
 * | Template   | is_fp32_dest_acc_en      | Whether DEST is in fp32 accumulation mode            | bool       |             | False    |
 * | Template   | block_rt_dim             | Height of the block in tiles (rows to pack)          | uint32_t   | 1           | False    |
 * | Function   | in                       | Input operand (tilized; unpack base)                 | LLKOperand |             | True     |
 * | Function   | out                      | Output operand (first row-major tile address)        | LLKOperand |             | True     |
 * | Function   | block_c_index            | Block column index (when full_ct_dim > block_ct_dim) | uint32_t   | >= 0        | False    |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    std::uint32_t block_rt_dim = 1,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_src_to_dest_block(
    LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out, std::uint32_t block_c_index = 0) {
    detail::pack_untilize_block<false, block_ct_dim, full_ct_dim, is_fp32_dest_acc_en, block_rt_dim>(
        in, out, block_c_index);
}

// clang-format off
/**
 * Id-free pack-untilize init, direct path (unpack straight into Dest). Takes an input and an output LLKOperand (data format + tile geometry as
 * NTTPs) instead of CB ids. Initializes all three threads (UNPACK/MATH/PACK) to move tilized tiles from the input
 * into DEST and pack them out row-major; register formats are derived inside the LLK. DEST remap is (re)configured
 * here. Input register format: 32-bit only (Float32 at fp32 Dest, Int32, UInt32) (compile-time checked). Blackhole only.
 *
 * | Param Type | Name                     | Description                               | Type       | Valid Range | Required |
 * |------------|--------------------------|-------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim/full_ct_dim | Block width / full input width in tiles   | uint32_t   |             | False    |
 * | Template   | is_fp32_dest_acc_en      | Whether DEST is in fp32 accumulation mode | bool       |             | False    |
 * | Function   | in                       | Input operand (tilized; format+geometry)  | LLKOperand |             | True     |
 * | Function   | out                      | Output operand (row-major destination)    | LLKOperand |             | True     |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_direct_to_dest_init(LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out) {
    detail::pack_untilize_init<true, block_ct_dim, full_ct_dim, is_fp32_dest_acc_en>(in, out);
}

// clang-format off
/**
 * Id-free pack-untilize of one block. Pair with pack_untilize_direct_to_dest_init. Owns the column loop and self-syncs DEST. Reads from
 * in.l1_address (unpack base; offset per column tile by InShape's stride) and writes the block's first row-major
 * output tile to out.l1_address. block_rt_dim must be 1 (no in/out row stride). Blackhole only.
 *
 * | Param Type | Name                     | Description                                          | Type       | Valid Range | Required |
 * |------------|--------------------------|------------------------------------------------------|------------|-------------|----------|
 * | Template   | block_ct_dim/full_ct_dim | Block width / full input width in tiles              | uint32_t   |             | False    |
 * | Template   | is_fp32_dest_acc_en      | Whether DEST is in fp32 accumulation mode            | bool       |             | False    |
 * | Template   | block_rt_dim             | Height of the block in tiles (rows to pack)          | uint32_t   | 1           | False    |
 * | Function   | in                       | Input operand (tilized; unpack base)                 | LLKOperand |             | True     |
 * | Function   | out                      | Output operand (first row-major tile address)        | LLKOperand |             | True     |
 * | Function   | block_c_index            | Block column index (when full_ct_dim > block_ct_dim) | uint32_t   | >= 0        | False    |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    std::uint32_t block_rt_dim = 1,
    DataFormat InFormat,
    TensorShape InShape,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_direct_to_dest_block(
    LLKOperand<InFormat, InShape> in, LLKOperand<OutFormat, OutShape> out, std::uint32_t block_c_index = 0) {
    detail::pack_untilize_block<true, block_ct_dim, full_ct_dim, is_fp32_dest_acc_en, block_rt_dim>(
        in, out, block_c_index);
}

// clang-format off
/**
 * Id-free pack-untilize DEST init (PACK thread only). Use when the tiles to untilize are already resident in
 * DEST (placed by copy_tile/reduce_tile/etc.) -- UNPACK and MATH are not configured here, unlike
 * pack_untilize_init. Takes only the output LLKOperand; the packer register format is derived inside the LLK
 * from OUT. Pair with pack_untilize_dest_block / pack_untilize_uninit. Blackhole only.
 *
 * | Param Type | Name          | Description                                       | Type       | Valid Range               | Required              |
 * |------------|---------------|---------------------------------------------------|------------|---------------------------|-----------------------|
 * | Template   | block_ct_dim  | Width of a single block in tiles                  | uint32_t   | 1 to max (DEST size)      | False (default = 8)   |
 * | Template   | full_ct_dim   | Width of a full input in tiles                    | uint32_t   | Divisible by block_ct_dim | False                 |
 * | Template   | diagonal      | Diagonal packing (unused on Blackhole; must be false) | bool   | false                     | False                 |
 * | Template   | narrow_row    | Whether the provided input is narrow              | bool       | true/false                | False                 |
 * | Template   | row_num_datums| Number of datums per row                          | uint32_t   | >= 1                      | False                 |
 * | Template   | dense         | Packs two 2-face tiles in a single 4-face region  | bool       | true/false                | False (default false) |
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode   | bool       |                           | False                 |
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
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_dest_init(LLKOperand<OutFormat, OutShape> /*out*/) {
    static_assert(diagonal == false, "pack_untilize_dest_init: diagonal is only supported on WH.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_dest_init: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_dest_init: full_ct_dim must be a positive multiple of block_ct_dim.");
    MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((llk_pack_untilize_init<
          LLKOperand<OutFormat, OutShape>::descriptor,
          is_fp32_dest_acc_en,
          block_ct_dim,
          full_ct_dim,
          narrow_row,
          row_num_datums,
          dense>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
}

// clang-format off
/**
 * Id-free pack-untilize of a block whose source is DEST (not an L1 operand). Packs (untilizes)
 * block_rt_dim tile-rows already in DEST (selected by tile_dst_ct_offset / tile_dst_rt_offset) out to the
 * row-major output at out.l1_address. Blackhole only.
 *
 * There is no input LLKOperand. The caller owns DEST synchronization (tile_regs_acquire/commit/wait/release).
 * Per-tile-row output stride is derived from the output descriptor (one tile's L1 size). Pair with
 * pack_untilize_dest_init.
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
 * | Template   | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode                     | bool       |                                         | False                 |
 * | Function   | out                | Output operand (first row-major output tile address)          | LLKOperand |                                         | True                  |
 * | Template   | block_rt_dim       | Height of the block in tiles (rows to pack)                    | uint32_t   | >= 1                                    | False (default = 1)   |
 * | Function   | block_c_index      | Block column index (used when full_ct_dim > block_ct_dim)      | uint32_t   | >= 0                                    | False (default = 0)   |
 * | Function   | tile_dst_rt_offset | Runtime offset of the tile index in DEST from which to pack    | uint32_t   | 0 to 7 (0 to 3 if fp32 dest enabled)    | False (default = 0)   |
 */
// clang-format on
template <
    std::uint32_t block_ct_dim = 8,
    std::uint32_t full_ct_dim = block_ct_dim,
    std::uint32_t block_rt_dim = 1,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    std::uint32_t tile_dst_ct_offset = 0,
    bool dense = false,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat OutFormat,
    TensorShape OutShape>
ALWI void pack_untilize_dest_block(
    LLKOperand<OutFormat, OutShape> out, std::uint32_t block_c_index = 0, std::uint32_t tile_dst_rt_offset = 0) {
    static_assert(diagonal == false, "pack_untilize_dest_block: diagonal is only supported on WH.");
    static_assert(is_legal_tile_shape(OutShape), "pack_untilize_dest_block: illegal output tile shape.");
    static_assert(
        block_ct_dim > 0 && full_ct_dim % block_ct_dim == 0,
        "pack_untilize_dest_block: full_ct_dim must be a positive multiple of block_ct_dim.");
    PACK((llk_pack_untilize<
          LLKOperand<OutFormat, OutShape>::descriptor,
          is_fp32_dest_acc_en,
          block_ct_dim,
          full_ct_dim,
          narrow_row,
          row_num_datums,
          tile_dst_ct_offset,
          dense>(block_rt_dim, out.l1_address, block_c_index, tile_dst_rt_offset)));
}

// clang-format off
/**
 * Id-free pack-untilize uninit. Restores the packer Z stride (via the output descriptor's register format)
 * and resets the packer to Default mode so a subsequent op can reprogram it. Blackhole only.
 *
 * | Template | is_fp32_dest_acc_en | Whether DEST is in fp32 accumulation mode | bool | | False |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat OutFormat, TensorShape OutShape>
ALWI void pack_untilize_uninit(LLKOperand<OutFormat, OutShape> /*out*/) {
    PACK((llk_pack_untilize_uninit<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((_llk_init_packer_dest_offset_registers_<DST_SYNC_MODE>()));
    PACK((llk_pack_reconfig_data_format<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en>()));
    PACK((llk_pack_init<LLKOperand<OutFormat, OutShape>::descriptor, is_fp32_dest_acc_en, PackMode::Default>()));
}

}  // namespace experimental
}  // namespace ckernel

#endif  // ARCH_QUASAR
