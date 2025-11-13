// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"

namespace ckernel {

// clang-format off
/**
 * Performs the necessary software initialization for the pack rows operation. This initialization
 * function configures the packer to pack a specified number of rows from the DEST register to L1.
 *
 * This function assumes the data in the DEST register is already in row-major format (not tilized).
 *
 * Key differences from pack_untilize:
 * - pack_untilize: expects tilized data in DEST, untilizes while packing
 * - pack_rows: expects row-major data in DEST, packs directly
 *
 * Current implementation uses a single packer interface for simplicity. Multi-packer support
 * can be added in future versions for improved performance.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                              | Type     | Valid Range | Required |
 * |------------|----------------|------------------------------------------|----------|-------------|----------|
 * | Template   | row_num_datums | Number of datums per row                 | uint32_t | >= 1        | False    |
 * | Function   | ocb            | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 * | Function   | num_rows       | Total number of rows to pack             | uint32_t | >= 1        | True     |
 */
// clang-format on
template <std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_rows_init(uint32_t ocb, const std::uint32_t num_rows) {
    PACK((llk_pack_rows_init<row_num_datums>(ocb, num_rows)));
}

// clang-format off
/**
 * Packs a specified number of rows from the DEST register to the output circular buffer (CB).
 * The data in the DEST register must already be in row-major format.
 *
 * This function must be preceded by a call to `pack_rows_init` to configure the packer for
 * row-based packing.
 *
 * | Param Type | Name           | Description                              | Type     | Valid Range                                          | Required |
 * |------------|----------------|------------------------------------------|----------|------------------------------------------------------|----------|
 * | Template   | row_num_datums | Number of datums per row                 | uint32_t | >= 1                                                 | False    |
 * | Function   | tile_index     | The index of the tile in the DEST register | uint32_t | Must be less than the size of the DEST register    | True     |
 * | Function   | ocb            | Output circular buffer identifier        | uint32_t | 0 to 31                                              | True     |
 */
// clang-format on
template <std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_rows(uint32_t tile_index, uint32_t ocb) {
    PACK((llk_pack_rows<row_num_datums>(tile_index, ocb)));
}

}  // namespace ckernel
