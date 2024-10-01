// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sfpu_types.h"
#include "llk_math_eltwise_unary_sfpu.h"

template <bool APPROXIMATE, class F, class ... ARGS>
inline void llk_math_eltwise_unary_sfpu_params(
    F&& sfpu_func,
    uint dst_index,
    int vector_mode = (int)VectorMode::RC,
    ARGS&& ... args) {

    _llk_math_eltwise_unary_sfpu_start_<DstSync::SyncHalf>(dst_index);

    if (vector_mode == (int)VectorMode::R) {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
           sfpu_func(static_cast<ARGS&&>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        // Skip next two faces
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    } else if (vector_mode == (int)VectorMode::C) {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    } else if (vector_mode == (int)VectorMode::RC) {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    } else {
        sfpu_func(static_cast<ARGS&&>(args)...);
    }
    _llk_math_eltwise_unary_sfpu_done_();
}
