// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_load_immediate_value.h"
#include "ckernel_sfpu_copy_values.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
/**
 * Loads an immediate floating point value into the specified DST register.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range |
 * Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst           | The index of the tile in DST register buffer to load the value into   | uint32_t | Must be less
 * than the size of the DST register buffer | True     | | val            | The floating point value to load | float |
 * Any valid floating point value                        | True     |
 */
ALWI void load_immediate_value(uint32_t idst, float val) {
    MATH(sfpu::llk_math_eltwise_unary_sfpu_load_imm(idst, val));
}

/**
 * Copies all values from the tile in idst1 to the tile in idst0 in the DST register buffer.
 *
 * The DST register buffer must be in acquired state via *tile_regs_acquire* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                           | Type     | Valid Range |
 * Required |
 * |----------------|-----------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | idst0          | The index of the tile in DST register buffer to copy values to        | uint32_t | Must be less
 * than the size of the DST register buffer | True     | | idst1          | The index of the tile in DST register buffer
 * to copy values from      | uint32_t | Must be less than the size of the DST register buffer | True     |
 */
ALWI void copy_values(uint idst0, uint32_t idst1) {
    MATH(sfpu::llk_math_eltwise_binary_sfpu_copy_values(idst0, idst1));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void load_immediate_value_init() {
    MATH(sfpu::llk_math_eltwise_unary_sfpu_load_imm_init());
}

ALWI void copy_values_init() {
    MATH(sfpu::llk_math_eltwise_binary_sfpu_copy_values_init());
}

}  // namespace ckernel
