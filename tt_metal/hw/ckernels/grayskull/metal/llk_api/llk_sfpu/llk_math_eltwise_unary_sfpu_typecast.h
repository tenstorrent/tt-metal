// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_typecast.h"

namespace ckernel {

// New LLK SFPU APIs

/// Convert from Float -> Int ///

/// Convert Float -> UInt16 ///
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_to_uint16_init() {
    //llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::to_uint16_tile_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_to_uint16_init(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                (ckernel::calculate_to_uint16<APPROXIMATE,4>,
				 ckernel::calculate_to_uint16<APPROXIMATE,4>,
				 dst_index, VectorMode::RC, param0);
}

/// Convert Float -> UInt32 ///
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_to_uint32_init() {
    //llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(ckernel::llk_math_calculate_sfpu::<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_to_uint32_init(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                (ckernel::calculate_to_uint32<APPROXIMATE,4>,
				 ckernel::calculate_to_uint32<APPROXIMATE,4>,
				 dst_index, VectorMode::RC, param0);
}

}
