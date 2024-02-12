// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "llk_math_eltwise_unary_sfpu_common_includes.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_typecast.h"

namespace ckernel {

// New LLK SFPU APIs

/// Convert from Float -> Int ///

/// Convert Float -> UInt16 ///
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_to_uint16_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(ckernel::sfpu::to_uint16_tile_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_to_uint16(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<SfpuType::to_uint16, APPROXIMATE, Dst>
                (ckernel::sfpu::calculate_to_uint16<APPROXIMATE,8>,
				 ckernel::sfpu::calculate_to_uint16<APPROXIMATE,8>,
				 dst_index, (int)VectorMode::RC);
}

/// Convert Float -> UInt32 ///
template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_to_uint32_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(ckernel::sfpu::to_uint32_tile_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_to_uint32(uint dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<SfpuType::to_uint32,APPROXIMATE, Dst>
                (ckernel::sfpu::calculate_to_uint32<APPROXIMATE,8>,
				 ckernel::sfpu::calculate_to_uint32<APPROXIMATE,8>,
				 dst_index, (int)VectorMode::RC);
}

}
