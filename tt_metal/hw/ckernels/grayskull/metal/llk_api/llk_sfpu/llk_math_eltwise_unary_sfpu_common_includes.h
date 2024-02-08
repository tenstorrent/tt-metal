// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "llk_sfpu_types.h"
#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_template.h"
#include "metal_ckernel_sfpu.h"
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
        _calculate_exponential_<APPROXIMATION_MODE, zero_negative, true, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::tanh) {
        _calculate_tanh_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::hardtanh) {
        _calculate_hardtanh_<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    } else if constexpr (operation == SfpuType::rsqrt) {
        //param0 = true -> approximate fast mode
        //         false -> high precision mode
        // The algorithm uses Newton's method based on no.of iteration better approximation can be calculated
        if ( param0 ) {
            calculate_rsqrt<true, ITERATIONS, 10>();
        } else {
            calculate_rsqrt<false, ITERATIONS, 25>();
        }
    } else if constexpr (operation == SfpuType::sigmoid) {
        calculate_sigmoid<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::sigmoid_appx) {
        calculate_sigmoid_appx<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::tanh_derivative) {
        _calculate_tanh_derivative_<APPROXIMATION_MODE, SfpuType_PARAM, ITERATIONS>();
    } else if constexpr (operation == SfpuType::dropout) {
        _calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(param0, param1);
    } else if constexpr (operation == SfpuType::power) {
        calculate_power_iterative<APPROXIMATION_MODE, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::square) {
        _calculate_square_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::log) {
        _calculate_log_<APPROXIMATION_MODE, false, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::log_with_base) {
        _calculate_log_<APPROXIMATION_MODE, true, ITERATIONS>(param0);
    } else if constexpr (
        (operation == SfpuType::equal_zero) ||
        (operation == SfpuType::not_equal_zero) ||
        (operation == SfpuType::less_than_zero) ||
        (operation == SfpuType::greater_than_equal_zero) ||
        (operation == SfpuType::less_than_equal_zero) ||
        (operation == SfpuType::greater_than_zero)) {
        calculate_comp<APPROXIMATION_MODE, operation, ITERATIONS>(8); //BFLOAT16 - exp
    } else if constexpr (operation == SfpuType::clamp) {
        _calculate_clamp_<APPROXIMATION_MODE, ITERATIONS>(param0, param1, param2);
    } else if constexpr (operation == SfpuType::abs) {
        _calculate_abs_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::sign) {
        calculate_sign<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::max) {
        _calculate_max_<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::min) {
        calculate_min<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::exp2) {
        calculate_exp2<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::heaviside) {
        calculate_heaviside<APPROXIMATION_MODE, ITERATIONS>(param0);
    } else if constexpr (operation == SfpuType::expm1) {
        calculate_expm1<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::asin) {
        calculate_asin<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::acos) {
        calculate_acos<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::atan) {
        calculate_atan<APPROXIMATION_MODE, ITERATIONS>();
    } else if constexpr (operation == SfpuType::signbit) {
        calculate_signbit<APPROXIMATION_MODE, ITERATIONS>();
    }else if constexpr (operation == SfpuType::to_uint16) {
        calculate_to_uint16<APPROXIMATION_MODE, ITERATIONS>();
    }else if constexpr (operation == SfpuType::to_uint32) {
        calculate_to_uint32<APPROXIMATION_MODE, ITERATIONS>();
    }else if constexpr (operation == SfpuType::silu) {
        calculate_silu<APPROXIMATION_MODE, ITERATIONS>();
    }
    //erf, erfc are dispatched directly.

}

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull, bool IS_INT_SFPU_EN=false>
inline void llk_math_eltwise_unary_sfpu(
    uint dst_index,
    int vector_mode = (int)VectorMode::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {

    _llk_math_eltwise_unary_sfpu_start_<Dst>(dst_index);
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
