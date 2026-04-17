// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_binary_sfpu_init.h"
#include "llk_math_eltwise_binary_sfpu_params.h"
#include "ckernel_sfpu_binary_comp.h"

namespace ckernel {

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_lt_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        llk_math_eltwise_binary_sfpu_init<SfpuType::lt>();
    } else {
        llk_math_eltwise_binary_sfpu_init<SfpuType::lt_uint>();
    }
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_lt_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for lt_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::lt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else if constexpr (DATA_FORMAT == DataFormat::UInt32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, ITERATIONS, SfpuType::lt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, ITERATIONS, SfpuType::lt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    }
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_gt_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for gt_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        llk_math_eltwise_binary_sfpu_init<SfpuType::gt>();
    } else {
        llk_math_eltwise_binary_sfpu_init<SfpuType::gt_uint>();
    }
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_gt_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for gt_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::gt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else if constexpr (DATA_FORMAT == DataFormat::UInt32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, ITERATIONS, SfpuType::gt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, ITERATIONS, SfpuType::gt>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    }
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_le_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        llk_math_eltwise_binary_sfpu_init<SfpuType::le>();
    } else {
        llk_math_eltwise_binary_sfpu_init<SfpuType::le_uint>();
    }
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_le_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for le_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::le>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else if constexpr (DATA_FORMAT == DataFormat::UInt32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, ITERATIONS, SfpuType::le>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, ITERATIONS, SfpuType::le>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    }
}

template <DataFormat DATA_FORMAT>
inline void llk_math_eltwise_binary_sfpu_ge_int_init() {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        llk_math_eltwise_binary_sfpu_init<SfpuType::ge>();
    } else {
        llk_math_eltwise_binary_sfpu_init<SfpuType::ge_uint>();
    }
}

template <bool APPROXIMATE, DataFormat DATA_FORMAT, int ITERATIONS = 8>
inline void llk_math_eltwise_binary_sfpu_ge_int(
    uint dst_index0, uint32_t dst_index1, uint32_t odst, int vector_mode = (int)VectorMode::RC) {
    static_assert(
        DATA_FORMAT == DataFormat::Int32 || DATA_FORMAT == DataFormat::UInt32 || DATA_FORMAT == DataFormat::UInt16,
        "Unsupported data format for ge_int. Supported data formats are: Int32, UInt32, UInt16");
    if constexpr (DATA_FORMAT == DataFormat::Int32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_int32<APPROXIMATE, ITERATIONS, SfpuType::ge>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else if constexpr (DATA_FORMAT == DataFormat::UInt32) {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint32<APPROXIMATE, ITERATIONS, SfpuType::ge>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    } else {
        _llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(
            ckernel::sfpu::calculate_binary_comp_uint16<APPROXIMATE, ITERATIONS, SfpuType::ge>,
            dst_index0,
            dst_index1,
            odst,
            vector_mode);
    }
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
