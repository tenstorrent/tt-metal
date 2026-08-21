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

#endif  // ARCH_BLACKHOLE

}  // namespace experimental
}  // namespace ckernel
