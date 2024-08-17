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

ALWI void unary_op_init_common(uint32_t icb, uint32_t ocb = 16)
{
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE>(icb) ));
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)  ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_init<false>(ocb) ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>() ));

    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb) ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));
    MATH(( llk_math_hw_configure_disaggregated() ));
}

ALWI void init_sfpu(uint32_t icb) {
    unary_op_init_common(icb);
}

} // namespace ckernel
