// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_comp.h"

namespace ckernel {

template <SfpuType OP, bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS>
inline void llk_math_eltwise_binary_sfpu_rel_int_impl(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format. Supported: Int32, UInt32, UInt16");
    constexpr auto sfpu_fn = []() {
        if constexpr (DATA_FORMAT == DataFormat::Int32) {
            return ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, OP>;
        } else {
            return ckernel::sfpu::calculate_binary_comp_uint<APPROXIMATE, ITERATIONS, OP, DATA_FORMAT>;
        }
    }();
    _llk_math_eltwise_binary_sfpu_params_(sfpu_fn, dst_index0, dst_index1, odst, vector_mode);
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_lt_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    llk_math_eltwise_binary_sfpu_init<SfpuType::lt_int>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_lt_int(uint i0, uint32_t i1, uint32_t o, int vm = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_rel_int_impl<SfpuType::lt, APPROXIMATE, DATA_FORMAT, ITERATIONS>(i0, i1, o, vm);
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_gt_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for gt_int. Supported data formats are: Int32, UInt32, UInt16");
    llk_math_eltwise_binary_sfpu_init<SfpuType::gt_int>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_gt_int(uint i0, uint32_t i1, uint32_t o, int vm = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_rel_int_impl<SfpuType::gt, APPROXIMATE, DATA_FORMAT, ITERATIONS>(i0, i1, o, vm);
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_le_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    llk_math_eltwise_binary_sfpu_init<SfpuType::le_int>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_le_int(uint i0, uint32_t i1, uint32_t o, int vm = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_rel_int_impl<SfpuType::le, APPROXIMATE, DATA_FORMAT, ITERATIONS>(i0, i1, o, vm);
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_ge_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    llk_math_eltwise_binary_sfpu_init<SfpuType::ge_int>();
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_ge_int(uint i0, uint32_t i1, uint32_t o, int vm = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu_rel_int_impl<SfpuType::ge, APPROXIMATE, DATA_FORMAT, ITERATIONS>(i0, i1, o, vm);
}

inline void llk_math_eltwise_binary_sfpu_eq_fp32_init() { llk_math_eltwise_binary_sfpu_init<SfpuType::eq>(); }

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
