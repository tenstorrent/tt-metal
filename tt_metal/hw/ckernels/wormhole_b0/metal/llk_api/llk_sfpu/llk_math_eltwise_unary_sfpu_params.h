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

    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    math::set_addr_mod_base();

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    if (vector_mode == (int)VectorMode::R) {
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
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
    } else if (vector_mode == (int)VectorMode::C) {
        // Do a column vector, Face0 + Face2 -- All iterations for full face
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else if (vector_mode == (int)VectorMode::RC) {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            sfpu_func(static_cast<ARGS&&>(args)...);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else {
        sfpu_func(static_cast<ARGS&&>(args)...);
    }
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}
