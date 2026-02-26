// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"
#include "api/compute/sentinel/compute_kernel_sentinel.h"

#ifdef TRISC_MATH
#include "experimental/llk_math_unary_datacopy_custom_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
namespace ckernel {

ALWI void copy_tile_custom(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
#ifdef ARCH_BLACKHOLE
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        0, true /*transpose within 16x16 face*/, in_cb_id)));
    UNPACK((llk_unpack_A(in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy_custom<A2D, DST_ACCUM_MODE, BroadcastType::NONE, false>(
        dst_tile_index, in_cb_id)));
#endif
}

}  // namespace ckernel
