#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_datacopy.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A.h"
#endif


namespace ckernel {

ALWI void transpose_wh_tile(uint32_t icb, uint32_t itile, uint32_t idst)
{
    UNPACK(( llk_unpack_A<BroadcastType::NONE, true>(icb, itile) ));

    MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(idst) ));
}

ALWI void transpose_wh_init(uint32_t icb)
{
    #ifdef ARCH_GRAYSKULL
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, true>() ));
    #else
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(true, true, icb) ));
    #endif

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, false>() ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<BroadcastType::NONE, true, true, false>(0) ));
    #else
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, true, EltwiseBinaryReuseDestType::NONE>(true, true)  ));
    UNPACK(( llk_unpack_A_hw_configure_disaggregated<>(0, true) ));
    #endif
}



} // namespace ckernel
