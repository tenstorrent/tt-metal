// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 //
 // SPDX-License-Identifier: Apache-2.0

 #pragma once

 #include "ckernel_sfpu_bitwise_or.h"
 #include "llk_math_eltwise_unary_sfpu_params.h"
 #include "llk_math_eltwise_unary_sfpu_init.h"

 namespace ckernel {

 // New LLK SFPU APIs

 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_bitwise_or_init() {
     llk_math_eltwise_unary_sfpu_init<SfpuType::bitwise_or, APPROXIMATE>();
 }

 template <bool APPROXIMATE>
 inline void llk_math_eltwise_unary_sfpu_bitwise_or(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
     llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
         ckernel::sfpu::calculate_bitwise_or<APPROXIMATE>,
         dst_index,
         vector_mode,
         param0);
 }

 }  // namespace ckernel
