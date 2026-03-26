// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#ifdef TRISC_PACK
#include "experimental/llk_pack_block_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Configures the packer for block-contiguous packing of tiny tiles.
 * Call once after pack initialization (or when tile dimensions change).
 * Does NOT need to be called again when only `num_tiles` changes.
 *
 * Precondition: `pack_tile` or other op-specific init must have been
 * called to set up the packer addr_mods and strides.
 *
 * Only available on Blackhole.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                                       | Type     | Valid Range | Required |
 * |------------|------|---------------------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void pack_block_contiguous_init(uint32_t ocb) {
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_block_contiguous_mop_config(ocb)));
#endif
}

// clang-format off
/**
 * Packs `num_tiles` tiles from the DEST register to a specified circular
 * buffer in a single call. Tiles are read from sparse Tile32x32 DEST
 * slots (standard convention) and written contiguously to L1.
 *
 * Before calling this function, `pack_block_contiguous_init` must have
 * been called to configure the MOP, and `cb_reserve_back(n)` must have
 * been called to reserve at least `num_tiles` tiles in the output CB.
 * The DEST register buffer must be in acquired state via *acquire_dst*.
 *
 * This call is blocking and is only available on the compute engine
 * (Blackhole only).
 *
 * Operates in tandem with functions cb_reserve_back and cb_push_back.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                       | Type     | Valid Range                                            | Required |
 * |------------|-----------|---------------------------------------------------|----------|--------------------------------------------------------|----------|
 * | Function   | ifrom_dst | The index of the first tile in the DEST register  | uint32_t | Must be less than the size of the DEST register (16)   | True     |
 * | Function   | ocb       | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                                | True     |
 * | Function   | num_tiles | Number of tiles to pack from DEST to CB           | uint32_t | 1 to 8                                                 | True     |
 */
// clang-format on
ALWI void pack_block_contiguous(uint32_t ifrom_dst, uint32_t ocb, uint32_t num_tiles) {
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_block_contiguous<DST_ACCUM_MODE>(ifrom_dst, ocb, num_tiles)));
#endif
}

}  // namespace ckernel
