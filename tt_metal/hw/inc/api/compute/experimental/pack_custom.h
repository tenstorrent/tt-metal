// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common_globals.h"

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
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false, false, false>(icb, num_tiles)));
#else
    (void)num_tiles;
    PACK((llk_pack_init<false, false, false>(icb)));
#endif
}

}  // namespace ckernel
