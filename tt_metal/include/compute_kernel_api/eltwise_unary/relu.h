/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_relu.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

//RELU MAX ops
ALWI void relu_max_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_relu_max<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_max_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_max_init<APPROX>() ));
}

//RELU MIN ops
ALWI void relu_min_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_relu_min<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void relu_min_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_min_init<APPROX>() ));
}

// RELU
ALWI void relu_tile(uint32_t idst) {
  MATH(( llk_math_eltwise_unary_sfpu_relu<APPROX, SyncHalf>(idst) ));
}

ALWI void relu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_relu_init<APPROX>() ));
}

//Leaky Relu : y = relu(x) + slope*-relu(-x)
ALWI void leaky_relu_tile(uint32_t idst,uint32_t param0) {
  MATH(( llk_math_eltwise_unary_sfpu_lrelu<APPROX, SyncHalf>(idst,param0) ));
}

ALWI void leaky_relu_tile_init() {
  MATH(( llk_math_eltwise_unary_sfpu_lrelu_init<APPROX>() ));
}

} // namespace ckernel
