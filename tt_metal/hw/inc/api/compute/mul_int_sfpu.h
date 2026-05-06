// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_mul_int.h"
#include "llk_math_eltwise_binary_sfpu_mul_int_replay.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise mul operation with the two integer inputs: y = mul(x0,x1)
 * Output overwrites odst in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * @tparam data_format Template argument specifying the data type.
 * Supported data formats are: DataFormat::Int32, DataFormat::UInt32, DataFormat::UInt16
 *
 * Return value: None
 *
 * | Argument              | Description                                                           | Type     | Valid Range                                           | Required |
 * |-----------------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0                 | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1                 | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | odst                  | The index of the tile in DST register buffer to use as output         | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
template <DataFormat data_format>
ALWI void mul_int_tile(uint32_t idst0, uint32_t idst1, uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_mul_int<APPROX, data_format>(idst0, idst1, odst)));
}

/**
 * Please refer to documentation for mul_int_tile.
 */
template <DataFormat data_format>
ALWI void mul_int_tile_init() {
    MATH((llk_math_eltwise_binary_sfpu_mul_int_init<APPROX, data_format>()));
}

// clang-format off
/**
 *  One-shot replay-buffer programming helpers. Must be called once per kernel,
 *  after `mul_int_tile_init<data_format>()`, on the MATH thread. The init
 *  configures sfpi prgm constants (LREG12-14) and addr_mods (ADDR_MOD_6/7);
 *  the helper records the body into replay slot 0 with `lltt::NoExec`.
 *
 *  `data_format` selects the body: UInt16 -> 8-bit-chunk discrete body,
 *  Int32 / UInt32 -> 11-bit-chunk fp32 body. UINT32 multiplication aliases
 *  INT32 because two's-complement multiplication of equal-width integers
 *  produces the same low 32 bits regardless of signedness.
*/
// clang-format on
template <DataFormat data_format>
ALWI void mul_int_binary_init_replay() {
    MATH((llk_math_eltwise_binary_sfpu_init_replay<data_format>()));
}
// clang-format off
/*
 * Drop-in replacement for `mul_int_tile<data_format>(idst0, idst1, odst)`.
 * The format-specific body is baked into the replay buffer by the matching
 * `mul_int_binary_init_replay<data_format>()`. Requires `idst1 == idst0 + 1`
 * and `odst == idst0`, matching the kernel's `(i*2, i*2 + 1, i*2)` pairing.
*/
// clang-format on
template <DataFormat data_format>
ALWI void mul_int_binary_tile_replay(std::uint32_t idst0, std::uint32_t idst1, std::uint32_t odst) {
    MATH((llk_math_eltwise_binary_sfpu_run_replay<data_format>(idst0, idst1, odst)));
}

}  // namespace ckernel
