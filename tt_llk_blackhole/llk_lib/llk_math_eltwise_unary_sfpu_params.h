// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <utility>

#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_sfpu_types.h"

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_unary_sfpu_params_(Callable&& sfpu_func, uint dst_index, int vector_mode = (int)VectorMode::RC, Args&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_<DST_SYNC_MODE>(dst_index);

    VectorMode mode = static_cast<VectorMode>(vector_mode);

    if (mode == VectorMode::R)
    {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
        // Skip next two faces
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
    }
    else if (mode == VectorMode::C)
    {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++)
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else if (mode == VectorMode::RC)
    {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++)
        {
            std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
            // Move to the next face
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        std::forward<Callable>(sfpu_func)(std::forward<Args>(args)...);
    }
    _llk_math_eltwise_unary_sfpu_done_();
}
