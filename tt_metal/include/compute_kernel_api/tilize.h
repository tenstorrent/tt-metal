// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"
#endif

#include "debug/dprint.h"


namespace ckernel {

/**
 * Initialize the tilize operation. To be called once at beginning of a kernel.
 */
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb = 16)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));

    MATH(( llk_math_pack_sync_init<SyncHalf>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_hw_configure_disaggregated<false>(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated(icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

/**
 * Re-initialize for the tilize operation. This can be called after a full init.
 */
ALWI void tilize_init_short(uint32_t icb, uint32_t block)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tilize_init_unpack(uint32_t icb, uint32_t block)
{
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

/**
 * Re-initialize for the tilize operation. This also reconfigure the unpacker with CB data type.
 */
ALWI void tilize_init_short_with_dt(uint32_t icb, uint32_t block) {

    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE>(0, 0, icb) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(1, 0) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

/**
 * Perform tilize operation on a block. This simply loops over the provided blocks.
 */
ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{
    UNPACK(( llk_unpack_tilize_block(icb, block) ));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH(( llk_math_wait_for_dest_available<SYNC>() ));
        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, SyncHalf>(0) ));
        PACK(( llk_pack<false, SYNC, false >(0, ocb)  ));

        // Release dest
        MATH(( llk_math_dest_section_done<SYNC>() ));
        PACK(( llk_pack_dest_section_done<SYNC>() ));
    }
}

ALWI void unpack_tilize_block(uint32_t icb, uint32_t block)
{
    UNPACK(( llk_unpack_tilize_block(icb, block) ));
}

/**
 * Uninitialize tilize operation before re-initializing for another operation.
 */
ALWI void tilize_uninit()
{
    UNPACK(( llk_unpack_tilize_uninit() ));
}

/**
 * Uninitialize the tilize operation along with re-configuring unpacker with the CB data types.
 */
ALWI void tilize_uninit_with_dt() {
    UNPACK(( llk_unpack_tilize_uninit() ));
    UNPACK(( llk_unpack_reconfig_data_format(0, 1, 0, 0) ));
}


}
