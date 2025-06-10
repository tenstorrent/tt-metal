// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_binary_sfpu_bitwise.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Performs an elementwise binary bitwise operation with the two inputs: y = bitwise(x0,x1)
 * Output overwrites first operand in DST.
 *
 * The DST register buffer must be in acquired state via *acquire_dst* call. This call is blocking and is only available
 * on the compute engine.
 * A maximum of 4 tiles from each operand can be loaded into DST at once, for a total of 8 tiles,
 * when using 16 bit formats. This gets reduced to 2 tiles from each operand for 32 bit formats.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range                                           | Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to use as first operand  | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | idst1          | The index of the tile in DST register buffer to use as second operand | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
// clang-format on
ALWI void bitwise_and_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::AND, InstrModLoadStore::INT32>(
        idst0, idst1)));
}

ALWI void bitwise_and_uint16_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::AND, InstrModLoadStore::LO16>(
        idst0, idst1)));
}

ALWI void bitwise_or_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::OR, InstrModLoadStore::INT32>(
        idst0, idst1)));
}

ALWI void bitwise_or_uint16_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::OR, InstrModLoadStore::LO16>(
        idst0, idst1)));
}

ALWI void bitwise_xor_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::XOR, InstrModLoadStore::INT32>(
        idst0, idst1)));
}

ALWI void bitwise_xor_uint16_binary_tile(uint32_t idst0, uint32_t idst1) {
    MATH((llk_math_eltwise_binary_sfpu_bitwise<APPROX, ckernel::sfpu::BinaryBitwiseOp::XOR, InstrModLoadStore::LO16>(
        idst0, idst1)));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void binary_bitwise_tile_init() { MATH((llk_math_eltwise_binary_sfpu_bitwise_init<APPROX>())); }

}  // namespace ckernel
