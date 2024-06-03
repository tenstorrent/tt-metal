// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_comp.h"

namespace ckernel {

// New LLK SFPU APIs

//EQZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_eqz(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::equal_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_eqz_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

//NEZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_nez(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::not_equal_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_nez_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

//LTZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_ltz(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::less_than_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_ltz_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

//GTZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gtz(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::greater_than_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gtz_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

//LEZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lez(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::less_than_equal_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_lez_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

//GEZ
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gez(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_comp<APPROXIMATE, SfpuType::greater_than_equal_zero>,
        dst_index,
        vector_mode,
        8);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gez_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>();
}

}
