// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_comp.h"

namespace ckernel {

inline void llk_math_eltwise_binary_sfpu_lt_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::lt>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::lt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_gt_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::gt>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::gt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_ge_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::ge>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::ge>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_le_int32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::le>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_int32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, 8, SfpuType::le>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::lt_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, 8, SfpuType::lt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::gt_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, 8, SfpuType::gt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::le_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, 8, SfpuType::le>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_uint32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::ge_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_uint32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, 8, SfpuType::ge>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_lt_uint16_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::lt_uint>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_uint16(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, 8, SfpuType::lt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_gt_uint16_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::gt_uint>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_uint16(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, 8, SfpuType::gt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_uint16_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::le_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_uint16(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, 8, SfpuType::le>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_uint16_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::ge_uint, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_uint16(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, 8, SfpuType::ge>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_eq_fp32_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::eq>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_eq_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::eq>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_ne_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::ne>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ne_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::ne>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_lt_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::lt>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_lt_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::lt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_gt_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::gt>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_gt_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::gt>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_le_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::le>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_le_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::le>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

inline void llk_math_eltwise_binary_sfpu_ge_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::ge>(); }

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_ge_fp32(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_binary_sfpu_params_(
        ckernel::sfpu::calculate_binary_comp_fp32<APPROXIMATE, 8, SfpuType::ge>,
        dst_index0,
        dst_index1,
        odst,
        vector_mode);
}

}  // namespace ckernel
