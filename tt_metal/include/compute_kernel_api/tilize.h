// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_reduce_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"
#endif

namespace ckernel {



/**
 * Initialize the tilize operation. To be called once at beginning of a kernel.
 */
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb = 16)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb) ));
    MATH(( llk_math_pack_sync_init<DST_ACCUM_MODE>() ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>(ocb) ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_tilize_hw_configure_disaggregated<DST_ACCUM_MODE>(icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)
/**
 * Initialize the tilize operation. To be called once at beginning of a kernel.
 */
ALWI void tilizeA_B_reduce_init(uint32_t icb0, uint32_t icb1_scaler, uint32_t block, uint32_t ocb = 16, uint32_t num_faces = 4, uint32_t face_r_dim = 16)
{
    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1_scaler) ));
    UNPACK(( llk_unpack_tilizeA_B_init<true, true>(icb0, icb1_scaler, block, num_faces, face_r_dim, 1) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init() ));

    PACK(( llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb) ));
    PACK(( llk_pack_init(ocb) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<false, DST_ACCUM_MODE>(ocb) ));
}
#endif

/**
 * Re-initialize for the tilize operation. This can be called after a full init.
 */
ALWI void tilize_init_short(uint32_t icb, uint32_t block)
{
    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb) ));
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tilize_init_unpack(uint32_t icb, uint32_t block)
{
    UNPACK(( llk_unpack_tilize_init(icb, block) ));
}

ALWI void tilizeA_B_init_unpack(uint32_t icb0, uint32_t icb1, uint32_t block)
{
    UNPACK(( llk_unpack_tilizeA_B_init(icb0, icb1, block) ));
}

/**
 * Re-initialize for the tilize operation. This also reconfigure the unpacker with CB data type.
 */
ALWI void tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block) {

    MATH(( llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, new_icb) ));
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    UNPACK(( llk_unpack_reconfig_data_format_srca(old_icb, new_icb) ));
    UNPACK(( llk_unpack_tilize_init(new_icb, block) ));
}

/**
 * Perform tilize operation on a block. This simply loops over the provided blocks.
 */
ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb)
{
    UNPACK(( llk_unpack_tilize_block(icb, block) ));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH(( llk_math_wait_for_dest_available() ));
        PACK(( llk_packer_wait_for_math_done() ));

        // Datacopy
        MATH(( llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE>(0 /*dst index*/) ));
        PACK(( llk_pack<false, false >(0 /*tile index*/, ocb)  ));

        // Release dest
        MATH(( llk_math_dest_section_done<DST_ACCUM_MODE>() ));
        PACK(( llk_pack_dest_section_done<DST_ACCUM_MODE>() ));
    }
}

ALWI void unpack_tilize_block(uint32_t icb, uint32_t block)
{
    UNPACK(( llk_unpack_tilize_block(icb, block) ));
}

ALWI void unpack_tilizeA_B_block(uint32_t icb0, uint32_t icb1, uint32_t block, uint32_t tile_idx_b, uint32_t num_faces = 4)
{
    UNPACK(( llk_unpack_tilizeA_B_block(icb0, icb1, block, tile_idx_b, num_faces) ));
}

/**
 * Uninitialize tilize operation before re-initializing for another operation.
 */
ALWI void tilize_uninit(uint32_t icb)
{
    UNPACK(( llk_unpack_tilize_uninit(icb) ));
}

/**
 * Uninitialize the tilize operation along with re-configuring unpacker with the CB data types.
 */
ALWI void tilize_uninit_with_dt(uint32_t old_icb = 0, uint32_t new_icb = 1) {
    UNPACK(( llk_unpack_tilize_uninit(old_icb) ));
    UNPACK(( llk_unpack_reconfig_data_format_srca(old_icb, new_icb) ));
}


}
