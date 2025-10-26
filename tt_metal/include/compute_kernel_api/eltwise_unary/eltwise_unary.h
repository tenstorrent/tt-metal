// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif

namespace ckernel {

ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE, StochRndType::None, true>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init<false>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>()));

    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(icb, icb)));
}

// clang-format off
/**
 * no_pack variant of unary_op_init_common, to be used with tilize_*_no_pack variants.
 *
 * Note: This function does not configure PACK thread, use with caution.
 */
// clang-format on
ALWI void unary_op_init_common_no_pack(uint32_t icb) {
    UNPACK((llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE, StochRndType::None, true>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));

    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    MATH((llk_math_hw_configure_disaggregated(icb, icb)));
}

ALWI void init_sfpu(uint32_t icb, uint32_t ocb) { unary_op_init_common(icb, ocb); }

}  // namespace ckernel
