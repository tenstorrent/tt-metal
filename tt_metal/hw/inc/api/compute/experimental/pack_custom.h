// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

namespace ckernel {

// clang-format off
/**
 * Initializes the packer for custom block packing operation. This function configures 
 * the packer to pack multiple tiles from the destination register bank to L1 memory.
 *
 * Call this function before using `pack_block_custom`. The function configures the packer
 * to handle the specified number of tiles (typically the full destination bank size).
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                       | Type     | Valid Range | Required |
 * |------------|-----------|---------------------------------------------------|----------|-------------|----------|
 * | Function   | icb       | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31     | True     |
 * | Function   | num_tiles | Number of tiles to configure the packer for       | uint32_t | 1 to 16     | True     |
 */
// clang-format on
ALWI void pack_block_init_custom(uint32_t icb, uint32_t num_tiles) {
    PACK((llk_pack_init<false, false, false>(icb, num_tiles)));
}

// clang-format off
/**
 * Packs a custom block of tiles from the destination register bank to the output circular buffer.
 * This function efficiently packs multiple tiles from the DEST register in a single operation.
 * 
 * Before calling this function:
 * 1. Initialize the pack block custom operation with `pack_block_init_custom`
 * 2. Ensure cb_reserve_back has been called on the output CB with sufficient space
 * 3. Data must be present in the destination register (from unpack/math operations)
 * 4. The DEST register buffer must be in acquired state via *acquire_dst* call
 *
 * Each call to `pack_block_custom` will pack the configured number of tiles (set via 
 * `pack_block_init_custom`) from the destination register to the output circular buffer.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                       | Type     | Valid Range                                          | Required |
 * |------------|-----------|---------------------------------------------------|----------|------------------------------------------------------|----------|
 * | Function   | ifrom_dst | The starting index in the DEST register           | uint32_t | Must be less than the size of the DEST register (16) | True     |
 * | Function   | icb       | The identifier of the output circular buffer (CB) | uint32_t | 0 to 31                                              | True     |
 * | Function   | ntiles    | The number of tiles to pack from DEST to CB       | uint32_t | 1 to 16                                              | True     |
 */
// clang-format on
ALWI void pack_block_custom(uint32_t ifrom_dst, uint32_t icb, uint32_t ntiles) {
    PACK((llk_matmul_pack<DST_ACCUM_MODE, false, false>(ifrom_dst, icb, ntiles)));
}

}  // namespace ckernel
