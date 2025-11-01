// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_ops.h"
#include "ckernel_sfpu.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"
#include "sfpu/ckernel_sfpu_welfords.h"

// local function declarations
inline void welfords_sfpu_configure_addrmod()
{
    addr_mod_t {
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_7);
}

inline void welfords_sfpu_configure_mop();

template <DstSync Dst>
inline void _llk_math_welfords_sfpu_start_(const uint dst_index)
{
    math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);

    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_welfords_sfpu_done_()
{
    math::clear_dst_reg_addr();

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0); // Equivalent to clear_addr_mod_base(): sets address modifier base for group 2 (addr mods 0..3) to 0.
                                                         // Parameter '2' selects the address modifier group, and '0' resets the base. This direct hardware
                                                         // instruction is used instead of the higher-level function for efficiency and explicit control.
}

inline void _llk_math_welfords_sfpu_inc_dst_face_addr_()
{
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_welfords_sfpu_init_()
{
    sfpu::_init_sfpu_config_reg();
    welfords_sfpu_configure_addrmod();
    math::reset_counters(p_setrwc::SET_ABD_F);
    lltt::record(0, 23);
    _compute_welfords_math_();      // 9 TTI instructions
    _welfords_load_initial_data_(); // 9 TTI instructions
    TTI_SFPTRANSP(0, 0, 0, 0);
    /*past_mean = dst1*/ TTI_SFPLOAD(p_sfpu::LREG4, 0, ADDR_MOD_3, 64);
    /*past_var = dst2*/ TTI_SFPLOAD(p_sfpu::LREG5, 0, ADDR_MOD_3, 128);
    // wiping LREG 6 and 7 since they may be filled with garbage data
    TTI_SFPLOADI(p_sfpu::LREG6, 0, 0);
    TTI_SFPLOADI(p_sfpu::LREG7, 0, 0);
}
