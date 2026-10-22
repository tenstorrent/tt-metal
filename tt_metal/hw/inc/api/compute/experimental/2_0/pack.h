// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef ARCH_QUASAR

#include <cstdint>
#include "api/compute/common_globals.h"
#include "api/compute/experimental/2_0/dest_sync.h"
#include "api/compute/experimental/2_0/llk_operand.h"
#include "experimental/2_0/llk_2_0.h"  // id-free Quasar LLK API

namespace ckernel {
namespace experimental {

// clang-format off
/**
 * Id-free pack init, Quasar. Pack TRISC programs a Pack0 buffer descriptor at out's base, the packer input format
 * (Float32 with a 32-bit Dest), no ReLU, L1 accumulation off, and the one-tile pack MOP. Joins the PACK unit to the
 * Dest chain. out's base is the DFB ring base, so one init serves every back() of the ring. Re-init when another
 * op ran in between (Quasar rule). No uninit. Only 16-bit float operands (Float16_b, Float16).
 * Pack buffer descriptors: the pack TRISC owns 8 (ids 16..23), re-allocated round robin by each pack_init; an op
 * only ever executes on the descriptor its own init programmed, so a kernel may call pack_init any number of times.
 *
 * | Param Type | Name                | Description                                       | Type        | Valid Range        | Required |
 * |------------|---------------------|---------------------------------------------------|-------------|--------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode                         | bool        |                    | False    |
 * | Template   | Format / Shape      | Output L1 format + geometry (deduced)             | DataFormat/TensorShape | Float16_b, Float16 | True |
 * | Function   | out                 | Output operand (e.g. DataflowBuffer::back)        | LLKOperand  | N/A                | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_init(LLKOperand<Format, Shape> out) {
    static_assert(llk_2_0::is_supported_format(Format), "pack_init (Quasar 2.0): only Float16_b / Float16.");
    static_assert(is_legal_tile_shape(Shape), "pack_init: illegal output tile shape.");
    detail::dest_join<dest_dvalid_client::PACK>();
    PACK((llk_2_0::pack_init<LLKOperand<Format, Shape>::descriptor, is_fp32_dest_acc_en>(out.l1_address)));
}

// clang-format off
/**
 * Id-free pack of one tile, Quasar: Dest tile `ifrom_dst` -> output tile `itile` of `out` (L1 tile
 * out.entry_idx + itile * out.stride_tiles). Takes Dest for PACK inside the section the kernel opened with
 * dest_acquire; PACK cannot be the first unit of a section.
 *
 * | Param Type | Name           | Description                                  | Type        | Valid Range            | Required |
 * |------------|----------------|----------------------------------------------|-------------|------------------------|----------|
 * | Template   | Format / Shape | Output L1 format + geometry (deduced)        | DataFormat/TensorShape | N/A   | True     |
 * | Function   | out            | Output operand (its base = the pack_init's)  | LLKOperand  | N/A                    | True     |
 * | Function   | itile          | Output tile index within out                 | uint32_t    | N/A                    | True     |
 * | Function   | ifrom_dst      | Dest tile index                              | uint32_t    | < Dest tiles / section | True     |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_tile(LLKOperand<Format, Shape> out, std::uint32_t itile, std::uint32_t ifrom_dst) {
    detail::dest_take<dest_dvalid_client::PACK>();
    PACK((llk_2_0::pack<LLKOperand<Format, Shape>::descriptor>(out.l1_address, ifrom_dst, out.tile(itile))));
}

// clang-format off
/**
 * Id-free block pack, Quasar: pack_tile over a ct_dim x rt_dim block, Dest tiles start_idst + i -> output tiles
 * start_itile + i, i < ct_dim * rt_dim. The block must fit one Dest section (static_assert).
 * No tails: every block op between two inits must cover the same tile count (LLK_ASSERT); for leftover tiles
 * re-init, then run the shorter block.
 *
 * | Param Type | Name                | Description                        | Type       | Valid Range                     | Required |
 * |------------|---------------------|------------------------------------|------------|---------------------------------|----------|
 * | Template   | ct_dim / rt_dim     | Block columns / rows (tiles)       | uint32_t   | ct_dim * rt_dim <= section size | True     |
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode          | bool       |                                 | False    |
 * | Function   | out                 | Output operand                     | LLKOperand | N/A                             | True     |
 * | Function   | start_itile         | First output tile within out       | uint32_t   | N/A                             | True     |
 * | Function   | start_idst          | First Dest tile                    | uint32_t   | N/A                             | False    |
 */
// clang-format on
template <
    std::uint32_t ct_dim,
    std::uint32_t rt_dim,
    bool is_fp32_dest_acc_en = DST_ACCUM_MODE,
    DataFormat Format,
    TensorShape Shape>
ALWI void pack_block(LLKOperand<Format, Shape> out, std::uint32_t start_itile, std::uint32_t start_idst = 0) {
    static_assert(
        ct_dim * rt_dim <= detail::dest_section_tiles<is_fp32_dest_acc_en>(), "pack_block: block exceeds Dest");
    pack_block<is_fp32_dest_acc_en>(out, start_itile, start_idst, ct_dim * rt_dim);
}

// clang-format off
/**
 * Id-free block pack with a runtime tile count, Quasar: pack_tile over `ntiles` tiles, Dest tiles
 * start_idst + i -> output tiles start_itile + i.
 * No tails: every block op between two inits must cover the same tile count (LLK_ASSERT); for leftover tiles
 * re-init, then run the shorter block.
 *
 * | Param Type | Name                | Description                        | Type       | Valid Range                         | Required |
 * |------------|---------------------|------------------------------------|------------|-------------------------------------|----------|
 * | Template   | is_fp32_dest_acc_en | fp32 dest-accumulate mode          | bool       |                                     | False    |
 * | Function   | out                 | Output operand                     | LLKOperand | N/A                                 | True     |
 * | Function   | start_itile         | First output tile within out       | uint32_t   | N/A                                 | True     |
 * | Function   | start_idst          | First Dest tile                    | uint32_t   | N/A                                 | True     |
 * | Function   | ntiles              | Number of tiles                    | uint32_t   | start_idst + ntiles <= section size | True     |
 */
// clang-format on
template <bool is_fp32_dest_acc_en = DST_ACCUM_MODE, DataFormat Format, TensorShape Shape>
ALWI void pack_block(
    LLKOperand<Format, Shape> out, std::uint32_t start_itile, std::uint32_t start_idst, std::uint32_t ntiles) {
    LLK_ASSERT(
        start_idst + ntiles <= detail::dest_section_tiles<is_fp32_dest_acc_en>(), "pack_block: block exceeds Dest");
    detail::check_block_tiles<dest_dvalid_client::PACK>(ntiles);
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        pack_tile(out, start_itile + i, start_idst + i);
    }
}

// clang-format off
/**
 * Packer L1 accumulation on/off, Quasar: the one runtime reconfig of the 2.0 API (pack_init turns it off). In a
 * loop over N partial results the first pack overwrites and packs 1..N-1 accumulate, so it toggles mid-op.
 *
 * | Param Type | Name   | Description                               | Type | Valid Range | Required |
 * |------------|--------|-------------------------------------------|------|-------------|----------|
 * | Function   | enable | true: packs add into L1; false: overwrite | bool | N/A         | True     |
 */
// clang-format on
ALWI void pack_reconfig_l1_acc(bool enable) { PACK((llk_2_0::pack_set_l1_acc(enable))); }

}  // namespace experimental
}  // namespace ckernel

#else  // Blackhole

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

#endif  // ARCH_QUASAR
