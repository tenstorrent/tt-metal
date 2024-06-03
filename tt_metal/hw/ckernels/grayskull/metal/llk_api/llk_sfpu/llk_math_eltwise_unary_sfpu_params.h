// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_eltwise_unary_sfpu_common_includes.h"

template <bool APPROXIMATE, class F, class ... ARGS>
inline void llk_math_eltwise_unary_sfpu_params(
    F&& sfpu_func,
    uint dst_index,
    int vector_mode = VectorMode::RC,
    ARGS&& ... args) {

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);

    if (vector_mode == VectorMode::R) {
        // Do a row vector, Face0 + Face1 -- first iteration
        const int ITERATIONS = 1;
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip the next 2 faces
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    } else if (vector_mode == VectorMode::C) {
        // Do a column vector, Face0 + Face2 -- full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else {
#pragma GCC unroll 0
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
        for (int face = 0; face < 4; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    math::clear_dst_reg_addr();
}
