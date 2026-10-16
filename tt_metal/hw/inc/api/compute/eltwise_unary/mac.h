// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_mac.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs elementwise multiply-accumulate (mac): out = a * b + c
 *
 * Must be called immediately after mac_tile_init(): the instruction sequence is
 * recorded into the replay buffer by mac_tile_init, and replay slots 0..6 are
 * shared with other SFPU ops. Any intervening SFPU op invalidates the recording.
 *
 * The tile indices below are fixed by that recording and cannot currently be
 * varied by the caller - operands are always read from tiles 0, 1, 2 and the
 * result is always written to tile 0. The arguments are retained for signature
 * compatibility with the other ternary SFPU ops.
 *
 * | Argument | Description                                              | Type     | Valid Range | Required |
 * |----------|----------------------------------------------------------|----------|-------------|----------|
 * | idst0    | Index of the tile in DST register buffer (input a)       | uint32_t | Must be 0   | True     |
 * | idst1    | Index of the tile in DST register buffer (input b)       | uint32_t | Must be 1   | True     |
 * | idst2    | Index of the tile in DST register buffer (input c)       | uint32_t | Must be 2   | True     |
 * | odst     | Index of the tile in DST register buffer (output)        | uint32_t | Must be 0   | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void mac_tile(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t idst2, std::uint32_t odst) {
    MATH((sfpu::Mac<APPROX, DST_ACCUM_MODE, data_format>::run(idst0, idst1, idst2, odst)));
}

/**
 * Please refer to documentation for any_init.
 */
template <DataFormat data_format>
ALWI void mac_tile_init() {
    MATH((sfpu::Mac<APPROX, DST_ACCUM_MODE, data_format>::init()));
}

}  // namespace ckernel
