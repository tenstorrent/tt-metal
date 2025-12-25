// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_max.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::max, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_max(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    _llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(
        ckernel::sfpu::calculate_max<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE, int ITERATIONS>
inline void llk_math_eltwise_unary_sfpu_max_iterations(
    uint dst_index, int vector_mode = (int)VectorMode::RC, int iterations = 1) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    ckernel::sfpu::_calculate_max_<APPROXIMATE, ITERATIONS>(iterations);

    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

}  // namespace ckernel
