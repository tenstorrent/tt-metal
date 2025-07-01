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

/**
 * Init function for untilize operations, to be used at the beginning of the kernel.
 */
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb) {
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated<true>(icb, icb)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim>(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, true>()));

    UNPACK((llk_unpack_A_hw_configure_disaggregated<DST_ACCUM_MODE>(icb)));
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false, false, icb)));  // init must be after configure
}

// clang-format off
/**
 * Perform the untilize operation on a block of tiles. This simply loops over the provided block size.
 * Performs the untilize operation on a block of tiles. Loops over the provided block size.
 *
 * | Param Type | Name          | Description                                 | Type      | Valid Range     | Required |
 * |------------|---------------|---------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim  | Width of a single block in tiles            | uint32_t  | 1 to 8          | False    |
 * | Template   | full_ct_dim   | Width of a full input in tiles              | uint32_t  | >= block_ct_dim | False    |
 * | Function   | icb           | Input circular buffer identifier            | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim  | Height of a single block in tiles           | uint32_t  | >= 1            | True     |
 * | Function   | ocb           | Output circular buffer identifier           | uint32_t  | 0 to 31         | True     |
 * | Function   | block_c_index | Index of the currently processed block      | uint32_t  | >= 0            | False    |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb, uint32_t block_c_index = 0) {
    for (uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            UNPACK(
                (llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(icb, c)));
            MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(c)));
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(1 /*num_blocks*/, ocb, FACE_R_DIM, 4, block_c_index)));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

/**
 * Uninitialize untilize operation, to allow initializing another operation.
 */
ALWI void pack_untilize_uninit(uint32_t ocb) {
    PACK((llk_pack_init(ocb)));
    PACK((llk_init_packer_dest_offset_registers<false>()));

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_untilize_uninit(ocb)));
#endif
}

template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dst_init_short(uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
    // A workaround for tt-metal#17132. Should be addressed more systematically.
    PACK((llk_pack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb, face_r_dim, num_faces)));
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
        ocb, face_r_dim, num_faces)));
    PACK((llk_init_packer_dest_offset_registers<true, diagonal>()));

    MATH((llk_math_hw_configure_disaggregated<true, true>(0, 0)));
}

template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dst(
    uint32_t ocb,
    uint32_t block_rt_dim = 1,
    uint32_t block_c_index = 0 /* used when full_ct_dim > block_ct_dim*/,
    uint32_t face_r_dim = 16,
    uint32_t num_faces = 4) {
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums>(
        block_rt_dim, ocb, face_r_dim, num_faces, block_c_index)));
}

template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init_short(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false, false, icb)));  // init must be after configure
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    pack_untilize_dst_init_short<block_ct_dim, full_ct_dim>(ocb);
}

}  // namespace ckernel
