// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(TRISC_PACK) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_pack_custom_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the packer for packing multiple contiguous tiles from the destination register bank to L1 memory.
 *
 * Call this function before using `pack_tile`. The function configures the packer
 * to handle the specified number of tiles (typically the full destination bank size).
 *
 * NOTE: This function alters the behavior of the packer only on Blachole.
 * Return value: None
 *
 * | Param Type | Name      | Description                                       | Type     | Valid Range | Required |
 * |------------|-----------|---------------------------------------------------|----------|-------------|----------|
 * | Function   | icb       | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | Function   | num_tiles | Number of tiles to configure the packer for       | uint32_t | 1 to 16     | True     |
 */
// clang-format on
ALWI void pack_block_init_custom(uint32_t icb, uint32_t num_tiles) {
// TODO NC: Should be removed as a custom functionality as a part of tt-metal#37671
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false, false, false>(icb, num_tiles)));
#else
    (void)num_tiles;
    PACK((llk_pack_init<false, false, false>(icb)));
#endif
}

// clang-format off
/**
 * Packs a single tile from the destination register to L1 memory without using the MOP
 * (Macro OPeration). This avoids MOP reprogramming overhead when alternating between
 * multi-tile and single-tile packing within the same kernel (e.g., packing a QKT block
 * vs. packing a single max/sum tile in SDPA).
 *
 * The DEST register buffer must be in acquired state via *acquire_dst* call. This call is
 * blocking and is only available on the compute engine. Before calling this function,
 * cb_reserve_back(n) must be called to reserve at least one tile in the output CB.
 *
 * Supports the same `out_of_order_output` mode as `pack_tile`.
 *
 * NOTE: Blackhole only. On other architectures falls back to regular `pack_tile`.
 * Requires that pack address modifiers (ADDR_MOD_0/1) are already configured by a prior
 * call to `pack_block_init_custom` or one of the standard pack init functions.
 *
 * Return value: None
 *
 * | Param Type | Name              | Description                                       | Type     | Valid Range                                          | Required |
 * |------------|-------------------|---------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Template   | out_of_order_output | When true, use output_tile_index for placement    | bool     | true or false                                        | False    |
 * | Function   | ifrom_dst         | The index of the tile in the DEST register        | uint32_t | Must be less than the size of the DEST register (16) | True     |
 * | Function   | icb               | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                              | True     |
 * | Function   | output_tile_index | The index of the tile in the output CB to copy to | uint32_t | Must be less than the size of the CB                 | False    |
 */
// clang-format on
template <bool out_of_order_output = false>
ALWI void pack_tile_no_mop(uint32_t ifrom_dst, uint32_t icb, std::uint32_t output_tile_index = 0) {
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_no_mop<DST_ACCUM_MODE, out_of_order_output>(ifrom_dst, icb, output_tile_index)));
#else
    PACK((llk_pack<DST_ACCUM_MODE, out_of_order_output, false>(ifrom_dst, icb, output_tile_index)));
#endif
}

}  // namespace ckernel
