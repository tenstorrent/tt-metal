// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary_sfpu.h"

/*************************************************************************
 * LLK ELTWISE BINARY SFPU
 *************************************************************************/

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu(
    const uint operand,
    uint dst_index_a,
    uint dst_index_b,
    int vector_mode = (int)VectorMode::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    const std::uint32_t operand_id = get_operand_id(0);
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const std::uint32_t face_r_dim = get_operand_face_r_dim(operand_id);

    _llk_math_eltwise_binary_sfpu_<sfpu_op, APPROXIMATE, DST_SYNC_MODE>(
        face_r_dim, num_faces, dst_index_a, dst_index_b, vector_mode, param0, param1, param2, param3, param4, param5);
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_init(
    uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    _llk_math_eltwise_binary_sfpu_init_<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::quant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_quant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::quant_int32, APPROXIMATE>(zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::requant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::requant_int32, APPROXIMATE>(zero_point);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32(
    uint dst_index_a, uint dst_index_b, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::dequant_int32, APPROXIMATE>(dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32_init(const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::dequant_int32, APPROXIMATE>(zero_point);
}
