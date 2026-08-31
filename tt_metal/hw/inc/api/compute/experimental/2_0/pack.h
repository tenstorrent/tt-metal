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

#ifdef ARCH_BLACKHOLE

// clang-format off
/**
 * Experimental id-free pack init. Takes an output LLKOperand (L1 format + geometry as NTTPs). The Dest
 * register format is derived INSIDE the LLK from the L1 format; the compute API never sees a register
 * format. Legacy ckernel::pack_init is untouched.
 *
 * | Template | Format | Output buffer L1 data format (deduced from LLKOperand) | DataFormat  |  | True |
 * | Template | Shape  | Output tile geometry (deduced from LLKOperand)         | TensorShape |  | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_init(LLKOperand<Format, Shape> /*out*/) {
    static_assert(is_legal_tile_shape(Shape), "pack_init: illegal output tile shape.");
    PACK((llk_pack_init<LLKOperand<Format, Shape>::descriptor, DST_ACCUM_MODE>()));
}

// clang-format off
/**
 * Experimental id-free pack. Copies one tile from DST to the absolute L1 address in the output LLKOperand.
 * Formats/geometry were programmed at pack_init; the pack op needs only the runtime write address
 * (out.l1_address, from the address seam) -- absolute (out-of-order) addressing. No id, no formats.
 *
 * DISCREPANCY vs legacy: legacy pack_tile(ifrom_dst, ocb) uses IN-ORDER packing (out_of_order=false) and
 * auto-advances the packer's internal fifo tile pointer, so consecutive calls write consecutive tiles. This
 * 2.0 pack uses OUT-OF-ORDER absolute addressing and does NOT advance any internal pointer: the caller must
 * supply the correct per-tile L1 address every call (e.g. cb_write_address(cb, t)). Equivalent to legacy for
 * the normal one-tile-per-reserve pattern; a caller relying on legacy auto-increment must instead index the
 * address itself.
 *
 * | Template | Format    | Output buffer L1 data format (deduced from LLKOperand) | DataFormat  |         | True |
 * | Template | Shape     | Output tile geometry (deduced from LLKOperand)         | TensorShape |         | True |
 * | Function | out       | The output L1 operand (format+shape+write address)    | LLKOperand  |         | True |
 * | Function | ifrom_dst | Tile index in the DST register                         | uint32_t   | 0 to 15 | True |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_tile(LLKOperand<Format, Shape> out, std::uint32_t ifrom_dst) {
    static_assert(is_legal_tile_shape(Shape), "pack_tile: illegal output tile shape.");
    // out_of_order_output=true: pack to the absolute address in the LLKOperand (no fifo_wr_tile_ptr bump).
    PACK((llk_pack<
          LLKOperand<Format, Shape>::descriptor,
          DST_ACCUM_MODE,
          /*out_of_order_output=*/true,
          PackMode::Default>(ifrom_dst, out.l1_address)));
}

// clang-format off
/**
 * Experimental id-free block pack. Packs `ntiles` consecutive tiles from the DST register to consecutive L1
 * addresses in the output operand -- the block/loop form of pack_tile. It is a thin compute-layer loop over the
 * 2.0 pack_tile: tile i is read from DST[ifrom_dst + i] and written to
 * out.l1_address + (start_out_tile + i) * tile_stride_words(Format, Shape), so the per-tile output stride is
 * derived from the COMPILE-TIME output tile geometry (internal/llk_descriptor.h), not from a CB page size. With
 * start_out_tile = 0 this reproduces exactly the L1 layout a legacy in-order pack_tile loop would write (each
 * consecutive call advancing by one tile). Each pack_tile uses absolute (out-of-order) addressing; there is no
 * internal fifo pointer, so the caller supplies the block base once and the op indexes from it. No id, no formats.
 *
 * | Param Type | Name          | Description                                                | Type        | Valid Range                          | Required |
 * |------------|---------------|------------------------------------------------------------|-------------|--------------------------------------|----------|
 * | Template   | Format        | Output buffer L1 data format (deduced from LLKOperand)     | DataFormat  |                                      | True     |
 * | Template   | Shape         | Output tile geometry (deduced from LLKOperand)            | TensorShape |                                      | True     |
 * | Function   | out           | The output L1 operand (format+shape+block base address)   | LLKOperand  |                                      | True     |
 * | Function   | ifrom_dst     | Index of the first tile in the DST register               | uint32_t    | 0 to 15                              | True     |
 * | Function   | ntiles        | Number of tiles to pack from DST to L1                     | uint32_t    | ifrom_dst + ntiles <= DST size (16) | True     |
 * | Function   | start_out_tile| Starting output tile index (offset into the block base)   | uint32_t    | N/A                                  | False    |
 */
// clang-format on
template <DataFormat Format, TensorShape Shape>
ALWI void pack_block(
    LLKOperand<Format, Shape> out, std::uint32_t ifrom_dst, std::uint32_t ntiles, std::uint32_t start_out_tile = 0) {
    static_assert(is_legal_tile_shape(Shape), "pack_block: illegal output tile shape.");
    for (std::uint32_t i = 0; i < ntiles; ++i) {
        // pack_tile is itself PACK()-wrapped, so the engine calls stay on the packer thread. Per-tile output
        // slot via the shared tile_address helper (stride folds to a compile-time constant; matches the CB
        // one-tile page).
        experimental::pack_tile(
            LLKOperand<Format, Shape>(detail::tile_address(out, start_out_tile + i)), ifrom_dst + i);
    }
}

// clang-format off
/**
 * Experimental id-free row pack. Packs a run of row-major rows (each 16 datums) from one DST tile to the
 * absolute L1 address in the output LLKOperand, in row-major (untilized) order -- the id-free successor to the
 * CB-id pack_rows(idst, ocb, output_index). The number of rows is configured beforehand by the (already
 * format-free / id-free) pack_rows_init(num_rows); this op needs only the DST tile index and the runtime write
 * address (out.l1_address, from the address seam -- e.g. cb_write_address(ocb, output_index)). Formats/geometry
 * were programmed at compute_kernel_hw_startup / pack_init; the row-pack HW path consumes no data format, so the
 * output LLKOperand supplies only the write address and the compile-time shape (for the legal-shape guard). No id.
 *
 * Pair with pack_rows_init / pack_rows_uninit (the CB-id-free legacy calls; both take no CB) exactly as the
 * legacy pack_rows does. Absolute (out-of-order) addressing: no internal fifo pointer is advanced, so the caller
 * supplies the correct per-output-region address each call.
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
    PACK((llk_pack_rows<LLKOperand<Format, Shape>::descriptor>(idst, out.l1_address)));
}

// clang-format off
/**
 * Experimental id-free row-pack init. Configures the packer to pack `num_rows` row-major rows (each 16 datums)
 * from a DST tile to L1 -- the id-free successor to the (already CB-id-free) legacy pack_rows_init(num_rows).
 * Takes the OUTPUT LLKOperand for API symmetry with the other 2.0 pack ops, but the row-pack init path programs
 * ONLY the packer counters/addrmods -- NO data format (formats come from compute_kernel_hw_startup / pack_init).
 * The operand therefore supplies only the compile-time shape (for the legal-shape guard); its address/format are
 * not read here. Reuses the existing format-free llk_pack_rows_init core (no new LLK op). Pair with pack_rows and
 * pack_rows_uninit. Legacy ckernel::pack_rows_init is untouched.
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
    PACK((llk_pack_rows_init(num_rows)));
}

// clang-format off
/**
 * Experimental id-free row-pack uninit. Restores the packer addrmods/counters to their default state after a run
 * of pack_rows, so subsequent standard packing (e.g. pack_tile) works -- the id-free successor to the (already
 * CB-id-free) legacy pack_rows_uninit(). Takes the OUTPUT LLKOperand for API symmetry only; the uninit path
 * programs no data format and reads nothing from the operand (the shape is used solely for the legal-shape guard).
 * Reuses the existing format-free llk_pack_rows_uninit core (no new LLK op). Legacy ckernel::pack_rows_uninit is
 * untouched.
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
    PACK((llk_pack_rows_uninit()));
}

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
