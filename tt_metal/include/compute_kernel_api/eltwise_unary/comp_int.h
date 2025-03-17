// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_comp_int.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

// clang-format off
 /**
  * Will store in the output of the compute core True if each element of a equal to zero.
  * The DST register buffer must be in acquired state via *acquire_dst* call.
  * This call is blocking and is only
  * available on the compute engine.
  *
  * Return value: None
  *
  * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
  * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
  * | idst            | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
  */
// clang-format on
ALWI void eqz_int_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_eqz_int<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void eqz_int_tile_init() { MATH((llk_math_eltwise_unary_sfpu_eqz_int_init<APPROX>())); }

}  // namespace ckernel
