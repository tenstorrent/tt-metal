// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "llk_sfpu_types.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_param_structs.h"
#include "llk_math_eltwise_unary_sfpu.h"

using namespace ckernel;
using namespace ckernel::sfpu;
namespace ckernel {

/*************************************************************************
 * LLK ELTWISE UNARY SFPU
 *************************************************************************/

template <
    SfpuType operation,
    bool APPROXIMATION_MODE,
    int SfpuType_PARAM = 0,
    int ITERATIONS = 4,
    bool IS_INT_SFPU_EN = false /*not used*/>
inline void llk_math_calculate_sfpu(
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    if constexpr (operation == SfpuType::exp_with_base) {
        constexpr bool zero_negative = true;
        _calculate_exponential_<APPROXIMATION_MODE, zero_negative, true, ITERATIONS>(ITERATIONS, param0);
    } else if constexpr (operation == SfpuType::tanh) {
        _calculate_tanh_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::hardtanh) {
        _calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    } else if constexpr (operation == SfpuType::tanh_derivative) {
        _calculate_tanh_derivative_<APPROXIMATION_MODE, SfpuType_PARAM, ITERATIONS>();
    } else if constexpr (operation == SfpuType::dropout) {
        _calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(param0, param1);
    } else if constexpr (operation == SfpuType::square) {
        _calculate_square_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::log) {
        _calculate_log_<APPROXIMATION_MODE, false, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::log_with_base) {
        _calculate_log_<APPROXIMATION_MODE, true, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::clamp) {
        _calculate_clamp_<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    } else if constexpr (operation == SfpuType::abs) {
        _calculate_abs_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::max) {
        _calculate_max_<APPROXIMATION_MODE, ITERATIONS>();
    }
    //erf, erfc are dispatched directly.

}

template <SfpuType sfpu_op, bool APPROXIMATE, bool IS_INT_SFPU_EN=false>
inline void llk_math_eltwise_unary_sfpu(
    uint dst_index,
    int vector_mode = (int)VectorMode::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {

    _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(dst_index);
    if (vector_mode == (int)VectorMode::R) {
        // Do a row vector, Face0 + Face1 -- first iteration
        const int ITERATIONS = 1;
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            llk_math_calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(param0, param1, param2, param3, param4, param5);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        // Skip the next 2 faces
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    } else if (vector_mode == (int)VectorMode::C) {
        // Do a column vector, Face0 + Face2 -- full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            llk_math_calculate_sfpu<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    } else {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            llk_math_calculate_sfpu<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    math::clear_dst_reg_addr();

}

}  // namespace ckernel
