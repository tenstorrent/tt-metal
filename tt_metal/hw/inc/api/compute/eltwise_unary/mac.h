// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

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
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mac_tile(uint32_t idst0, uint32_t idst1, uint32_t idst2, uint32_t odst) {
    MATH((sfpu::Mac<APPROX, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::calculate(
        idst0, idst1, idst2, odst, VectorMode::RC)));
}

/**
 * Please refer to documentation for any_init.
 */
template <DataFormat data_format, bool is_fp32_dest_acc_en = DST_ACCUM_MODE>
ALWI void mac_tile_init() {
    MATH((sfpu::Mac<APPROX, data_format, DST_SYNC_MODE, is_fp32_dest_acc_en>::init()));
}

}  // namespace ckernel
