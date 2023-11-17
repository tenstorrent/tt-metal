// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#endif



namespace ckernel {

ALWI void unary_op_init_common(uint32_t icb)
{
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE>() ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE>(icb) ));
    #else
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE>()  ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<>(icb) ));
    #endif

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));

    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, false>() ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    #endif
    MATH(( llk_math_pack_sync_init<SYNC>() ));
}

ALWI void init_sfpu(uint32_t icb) {
    unary_op_init_common(icb);
}

} // namespace ckernel
