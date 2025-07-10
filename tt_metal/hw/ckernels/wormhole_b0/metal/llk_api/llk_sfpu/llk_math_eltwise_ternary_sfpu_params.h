// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_sfpu_types.h"
#include "llk_math_eltwise_ternary_sfpu.h"
#include "ckernel_sfpu_where.h"

template <bool APPROXIMATE, class F, class... ARGS>
inline void llk_math_eltwise_ternary_sfpu_params(
    F&& sfpu_func,
    uint dst_index0,
    uint dst_index1,
    uint dst_index2,
    int vector_mode = (int)VectorMode::RC,
    ARGS&&... args) {
    // Compute minimum destination index
    uint dst_index = std::min(std::min(dst_index0, dst_index1), dst_index2);

    // Compute max difference among indices to determine offset
    uint dst_offset = std::max(std::max(dst_index0, dst_index1), dst_index2) - dst_index;

    _llk_math_eltwise_ternary_sfpu_start_<DST_SYNC_MODE>(dst_index);  // Reuse same sync primitive

    if (vector_mode == (int)VectorMode::R) {
        // Row vector - Face0 + Face1
        for (int face = 0; face < 2; face++) {
            ckernel::sfpu::calculate_where_fp32<APPROXIMATE>();
            // sfpu_func(static_cast<ARGS&&>(args)...); //Need to replace the above line with this
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);  // repeat 2x
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        // Skip next 2 faces
        for (int i = 0; i < 4; ++i) {
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }

    } else if (vector_mode == (int)VectorMode::C) {
        // Column vector - Face0 + Face2
        for (int face = 0; face < 2; face++) {
            ckernel::sfpu::calculate_where_fp32<APPROXIMATE>();
            // sfpu_func(dst_offset, static_cast<ARGS&&>(args)...); //Need to replace the above line with this
            for (int i = 0; i < 4; ++i) {
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }

    } else if (vector_mode == (int)VectorMode::RC) {
        // All 4 faces
        for (int face = 0; face < 4; face++) {
            ckernel::sfpu::calculate_where_fp32<APPROXIMATE>();
            // sfpu_func(dst_offset, static_cast<ARGS&&>(args)...); //Need to replace the above line with this
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }

    } else {
        // Default: single face pass-through
        ckernel::sfpu::calculate_where_fp32<APPROXIMATE>();
        // sfpu_func(dst_offset, static_cast<ARGS&&>(args)...); //Need to replace the above line with this
    }

    _llk_math_eltwise_ternary_sfpu_done_();  // Finalize
}
