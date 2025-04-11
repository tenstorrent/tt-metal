// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_reduce.h"

/*************************************************************************
 * LLK REDUCE
 *************************************************************************/

template <
    PoolType type,
    ReduceDim dim,
    int num_fidelity_phases,
    bool is_fp32_dest_acc_en,
    bool is_int_fpu_en>
inline void llk_math_reduce(const uint dst_index, const uint num_faces) {
    _llk_math_reduce_<type, dim, num_fidelity_phases, is_fp32_dest_acc_en, is_int_fpu_en>(dst_index, false, num_faces);
}

template <PoolType type, ReduceDim dim, int num_fidelity_phases>
inline void llk_math_reduce_init(const std::uint32_t within_face_16x16_transpose) {
    _llk_math_reduce_init_<type, dim, num_fidelity_phases>(within_face_16x16_transpose);
}
