// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_comp.h"
#include "llk_math_eltwise_unary_sfpu_exp2.h"
#include "llk_math_eltwise_unary_sfpu_expm1.h"
#include "llk_math_eltwise_unary_sfpu_heaviside.h"
#include "llk_math_eltwise_unary_sfpu_power.h"
#include "llk_math_eltwise_unary_sfpu_rsqrt.h"
#include "llk_math_eltwise_unary_sfpu_tiled_prod.h"
#include "llk_math_eltwise_unary_sfpu_sigmoid.h"
#include "llk_math_eltwise_unary_sfpu_sign.h"
#include "llk_math_eltwise_unary_sfpu_signbit.h"
#include "llk_math_eltwise_unary_sfpu_silu.h"
#include "llk_math_eltwise_unary_sfpu_topk.h"
#include "llk_math_eltwise_unary_sfpu_trigonometry.h"
#include "llk_math_eltwise_unary_sfpu_unary_comp.h"
#include "llk_math_eltwise_unary_sfpu_zeros.h"

namespace ckernel {

/*************************************************************************
* LLK ELTWISE UNARY SFPU
*************************************************************************/

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::log, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>();
}

//abs
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::abs, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>();
}

//log with base
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base(uint dst_index, uint base, int vector_mode = VectorMode::RC) {
  llk_math_eltwise_unary_sfpu<SfpuType::log_with_base, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode, base);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::tanh, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>();
}

inline void llk_math_eltwise_unary_sfpu_dropout(
    uint dst_index, int vector_mode, int integer_dropout, int scale_factor) {
    constexpr bool dont_care = false;
    llk_math_eltwise_unary_sfpu<SfpuType::dropout, dont_care, DstSync::SyncHalf>(
        dst_index, vector_mode, integer_dropout, scale_factor);
}

inline void llk_math_eltwise_unary_sfpu_dropout_init(uint seed = 0) {
    constexpr bool dont_care = false;
    constexpr uint dont_care_param = 0;

    llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, dont_care>(dont_care_param, dont_care_param, seed);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max(uint dst_index, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::max, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::max, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_square(uint dst_index, int vector_mode = VectorMode::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::square, APPROXIMATE, DstSync::SyncHalf>(dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_square_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::square, APPROXIMATE>();
}

//ELU - implemented in ckernel_sfpu_elu.h

}
