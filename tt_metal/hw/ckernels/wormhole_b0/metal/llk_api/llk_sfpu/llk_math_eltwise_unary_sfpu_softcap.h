// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Self-contained LLK wrapper that avoids the broken ckernel_sfpu.h include chain.
// We inline the tile-iteration boilerplate from _llk_math_eltwise_unary_sfpu_params_
// to avoid transitive includes of nuked headers.

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_softcap.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_softcap_init() {
    sfpu::_init_sfpu_config_reg();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

template <bool APPROXIMATE, int ITERATIONS = 8>
inline void llk_math_eltwise_unary_sfpu_softcap(
    uint dst_index, uint32_t param0, int vector_mode = (int)VectorMode::RC) {
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(dst_index);
    math::set_addr_mod_base();

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);

    if (vector_mode == (int)VectorMode::RC) {
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            ckernel::sfpu::calculate_softcap<APPROXIMATE, ITERATIONS>(param0);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    } else {
        ckernel::sfpu::calculate_softcap<APPROXIMATE, ITERATIONS>(param0);
    }

    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

}  // namespace ckernel
