// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common_globals.h"

#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_custom_api.h"
#endif

#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

#ifdef ARCH_BLACKHOLE
ALWI void copy_tile_custom(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index) {
    UNPACK((llk_unpack_A_custom<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        in_cb_id, in_tile_index)));
    MATH((llk_math_eltwise_unary_datacopy_custom(dst_tile_index)));
}
#endif

}  // namespace ckernel
