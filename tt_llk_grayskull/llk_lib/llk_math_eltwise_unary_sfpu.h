// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"

using namespace ckernel;

template <DstSync Dst>
inline void _llk_math_eltwise_unary_sfpu_start_(const uint dst_index) {
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
}

inline void _llk_math_eltwise_unary_sfpu_done_() {
    math::clear_dst_reg_addr();
}

inline void _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_() {
    math::inc_dst_addr<8>();
    math::inc_dst_addr<8>();
}

inline void _llk_math_eltwise_unary_sfpu_init_() {
    math::reset_counters(p_setrwc::SET_ABD_F);
}
