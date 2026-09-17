// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/llk_operand.h"

#ifdef TRISC_PACK
#include "experimental/2_0/llk_pack_tile.h"
#endif

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free pack init. Takes an output LLKOperand (L1 format + geometry as NTTPs); the DST register format is
 * derived inside the LLK from the L1 format. Blackhole only.
 *
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        |  | False |
 * | Template | Format              | Output buffer L1 data format (deduced from LLKOperand) | DataFormat  |  | True |
 * | Template | Shape               | Output tile geometry (deduced from LLKOperand)         | TensorShape |  | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_init(LLKOperand<Format, Shape> /*out*/) {
    static_assert(is_legal_tile_shape(Shape), "pack_init: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    PACK((llk_pack_init<LLKOperand<Format, Shape>::descriptor, is_fp32_dest_acc_en>()));
}

// clang-format off
/**
 * Id-free pack. Copies one tile from DST to L1. `out.l1_address` is the buffer base; the pack address is
 * `tile_address(out, itile)`. Formats and geometry were programmed at pack_init. Blackhole only.
 *
 * Uses out-of-order (absolute) addressing and does not auto-advance an internal fifo pointer like legacy
 * pack_tile does. Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Template | is_fp32_dest_acc_en | fp32 dest-accumulate mode                          | bool        |         | False |
 * | Template | Format    | Output buffer L1 data format (deduced from LLKOperand) | DataFormat  |         | True |
 * | Template | Shape     | Output tile geometry (deduced from LLKOperand)         | TensorShape |         | True |
 * | Function | out       | The output L1 operand (format+shape+buffer base)      | LLKOperand  |         | True |
 * | Function | itile     | Index of the output tile within `out`, relative to its base | uint32_t | N/A   | True |
 * | Function | ifrom_dst | Tile index in the DST register                         | uint32_t   | 0 to 15 | True |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_tile(LLKOperand<Format, Shape> out, std::uint32_t itile, std::uint32_t ifrom_dst) {
    static_assert(is_legal_tile_shape(Shape), "pack_tile: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    // out_of_order_output=true: pack to the absolute address (no fifo_wr_tile_ptr bump).
    PACK((llk_pack<
          LLKOperand<Format, Shape>::descriptor,
          is_fp32_dest_acc_en,
          /*out_of_order_output=*/true,
          PackMode::Default>(ifrom_dst, detail::tile_address(out, itile))));
}

// clang-format off
/**
 * Id-free block pack. Packs `ntiles` consecutive tiles from DST to consecutive L1 tiles in the output
 * operand (block/loop form of pack_tile). Tile i is read from DST[ifrom_dst + i] and written to
 * output tile (start_out_tile + i). Blackhole only. Sub-32-row (partial-height) block-float tiles are not
 * supported (compile-time rejected).
 *
 * | Param Type | Name          | Description                                                | Type        | Valid Range                          | Required |
 * |------------|---------------|------------------------------------------------------------|-------------|--------------------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                             | bool        |                                      | False    |
 * | Template   | Format        | Output buffer L1 data format (deduced from LLKOperand)     | DataFormat  |                                      | True     |
 * | Template   | Shape         | Output tile geometry (deduced from LLKOperand)            | TensorShape |                                      | True     |
 * | Function   | out           | The output L1 operand (format+shape+block base address)   | LLKOperand  |                                      | True     |
 * | Function   | ifrom_dst     | Index of the first tile in the DST register               | uint32_t    | 0 to 15                              | True     |
 * | Function   | ntiles        | Number of tiles to pack from DST to L1                     | uint32_t    | ifrom_dst + ntiles <= DST size (16) | True     |
 * | Function   | start_out_tile| Starting output tile index (offset into the block base)   | uint32_t    | N/A                                  | False    |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_block(
    LLKOperand<Format, Shape> out, std::uint32_t ifrom_dst, std::uint32_t ntiles, std::uint32_t start_out_tile = 0) {
    static_assert(is_legal_tile_shape(Shape), "pack_block: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        experimental::pack_tile<is_fp32_dest_acc_en>(out, start_out_tile + i, ifrom_dst + i);
    }
}

// clang-format off
/**
 * Id-free row pack. Packs a run of row-major rows (each 16 datums) from one DST tile to the absolute L1
 * address in the output LLKOperand, in row-major (untilized) order. Blackhole only.
 *
 * The number of rows is configured beforehand by pack_rows_init(num_rows). Uses out-of-order (absolute)
 * addressing -- no internal fifo pointer is advanced, so the caller supplies the correct address each call.
 * Pair with pack_rows_init / pack_rows_uninit. Sub-32-row (partial-height) block-float tiles are not
 * supported (compile-time rejected).
 *
 * | Param Type | Name  | Description                                              | Type        | Valid Range                          | Required |
 * |------------|-------|----------------------------------------------------------|-------------|--------------------------------------|----------|
 * | Template   | Format| Output buffer L1 data format (deduced from LLKOperand)   | DataFormat  |                                      | True     |
 * | Template   | Shape | Output tile geometry (deduced from LLKOperand)          | TensorShape |                                      | True     |
 * | Function   | out   | The output L1 operand (format+shape+write address)      | LLKOperand  |                                      | True     |
 * | Function   | idst  | Index of the tile in the DST register to pack rows from | uint32_t    | 0 to 15                              | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_rows(LLKOperand<Format, Shape> out, std::uint32_t idst) {
    static_assert(is_legal_tile_shape(Shape), "pack_rows: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    PACK((llk_pack_rows<LLKOperand<Format, Shape>::descriptor>(idst, out.l1_address)));
}

// clang-format off
/**
 * Id-free row-pack init. Configures the packer to pack `num_rows` row-major rows (each 16 datums) from a DST
 * tile to L1. Takes the output LLKOperand for API symmetry with the other 2.0 pack ops, but programs only the
 * packer counters/addrmods -- no data format (formats come from compute_kernel_hw_startup / pack_init); the
 * operand supplies only the compile-time shape (legal-shape guard). Pair with pack_rows and pack_rows_uninit.
 * Blackhole only.
 *
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Param Type | Name     | Description                                                   | Type        | Valid Range | Required |
 * |------------|----------|---------------------------------------------------------------|-------------|-------------|----------|
 * | Template   | Format   | Output buffer L1 data format (deduced from LLKOperand)        | DataFormat  |             | True     |
 * | Template   | Shape    | Output tile geometry (deduced from LLKOperand)               | TensorShape |             | True     |
 * | Function   | out      | The output L1 operand (used only for the compile-time shape)  | LLKOperand  |             | True     |
 * | Function   | num_rows | Number of rows to pack from DST to L1 (each row = 16 datums)  | uint32_t    | 1 to 64     | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_rows_init(LLKOperand<Format, Shape> /*out*/, std::uint32_t num_rows) {
    static_assert(is_legal_tile_shape(Shape), "pack_rows_init: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    PACK((llk_pack_rows_init(num_rows)));
}

// clang-format off
/**
 * Id-free row-pack uninit. Restores the packer addrmods/counters to their default state after a run of
 * pack_rows, so subsequent standard packing (e.g. pack_tile) works. Takes the output LLKOperand for API
 * symmetry only; programs no data format and reads nothing from the operand (the shape is used solely for
 * the legal-shape guard). Blackhole only.
 *
 * Sub-32-row (partial-height) block-float tiles are not supported (compile-time rejected).
 *
 * | Param Type | Name | Description                                                  | Type        | Valid Range | Required |
 * |------------|------|--------------------------------------------------------------|-------------|-------------|----------|
 * | Template   | Format | Output buffer L1 data format (deduced from LLKOperand)      | DataFormat  |             | True     |
 * | Template   | Shape  | Output tile geometry (deduced from LLKOperand)             | TensorShape |             | True     |
 * | Function   | out    | The output L1 operand (used only for the compile-time shape) | LLKOperand  |             | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_rows_uninit(LLKOperand<Format, Shape> /*out*/) {
    static_assert(is_legal_tile_shape(Shape), "pack_rows_uninit: illegal output tile shape.");
    static_assert(
        !(is_block_float_format(Format) && is_partial_height(Shape)),
        "pack: sub-32-row (partial-height) block-float tiles are not supported on the BH compute datapath; "
        "use a full 32-row tile.");
    PACK((llk_pack_rows_uninit()));
}

}  // namespace experimental
}  // namespace ckernel
