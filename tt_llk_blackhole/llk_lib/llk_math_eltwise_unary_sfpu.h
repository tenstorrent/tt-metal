// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#pragma once
#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_sfpu_types.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"

using namespace ckernel;
// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_unary_sfpu_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);

    if (sfpu_op == SfpuType::topk_local_sort) {
        addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 32},
        }.set(ADDR_MOD_6);
    }
}
inline void eltwise_unary_sfpu_configure_mop();

template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const uint dst_index) {
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
}

inline void _llk_math_eltwise_unary_sfpu_done_() {
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_() {
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_() {
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpu_configure_addrmod<sfpu_op>();
    math::reset_counters(p_setrwc::SET_ABD_F);
}
