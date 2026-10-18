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
 * | Function | ifrom_dst | Tile index in the DST register                         | uint32_t   | Must be less than the acquired size of DST REG | True |
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
 * | Function   | ifrom_dst     | Index of the first tile in the DST register               | uint32_t    | Must be less than the acquired size of DST REG | True     |
 * | Function   | ntiles        | Number of tiles to pack from DST to L1                     | uint32_t    | ifrom_dst + ntiles <= acquired size of DST REG | True     |
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

}  // namespace experimental
}  // namespace ckernel
