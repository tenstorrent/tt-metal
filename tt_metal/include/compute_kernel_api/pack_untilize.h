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

// clang-format off
/**
 * Performs the necessary hardware and software initialization for the pack untilize operation. This initialization
 * function should be used when the desired PACK input is already in DEST register - therefore, it doesn't
 * configure UNPACK and MATH threads for transferring data from circular buffers to DEST register
 *
 * This function allows the user to specify face_r_dim and num_faces through function parameters. Setting these
 * parameters results in an expensive MMIO write, so it should be used only when necessary.
 * This should addressed more systematically within the issue tt-metal#22820, since these two values can be inferred
 * from the circular buffer description, the same way as it is done in llk_pack_hw_configure_disaggregated. This
 * would remove the need for llk_pack_untilize_hw_configure_disaggregated altogether and we would pay the price
 * of the MMIO write only once, in compute_kernel_hw_startup.
 *
 * | Param Type | Name         | Description                              | Type      | Valid Range     | Required |
 * |------------|--------------|------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles         | uint32_t  | 1 to 8          | False    |
 * | Template   | full_ct_dim  | Width of a full input in tiles           | uint32_t  | >= block_ct_dim | False    |
 * | Template   | narrow_row   |  Whether the provided input is narrow    | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row               | uint32_t  | >= 1            | False    |
 * | Function   | ocb          | Output circular buffer identifier        | uint32_t  | 0 to 31         | True     |
 * | Function   | face_r_dim   | Face height in rows                      | uint32_t  | 1, 8 or 16      | False    |
 * | Function   | num_faces    | Number of faces                          | uint32_t  | 1, 2 or 4       | False    |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dst_init(uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
#ifdef ARCH_BLACKHOLE
    // Needed for setting swizzle_32b:
    MATH((llk_math_hw_configure_disaggregated<false, true>(0, 0)));
#endif
    // A workaround for tt-metal#17132. Should be addressed more systematically.
    PACK(
        (llk_pack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE, false /*untilize*/>(ocb, face_r_dim, num_faces)));
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, false, narrow_row, row_num_datums>(
        ocb, face_r_dim, num_faces)));
    PACK((llk_init_packer_dest_offset_registers<true, false>()));
}

// clang-format off
/**
 * Performs the necessary hardware and software initialization for the pack untilize operation. Initializes all three
 * threads: UNPACK, MATH, and PACK. This function should be used when the desired PACK input is not yet in DEST
 * register.
 *
 * | Param Type | Name         | Description                                | Type      | Valid Range     | Required |
 * |------------|--------------|--------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles           | uint32_t  | 1 to 8          | False    |
 * | Template   | full_ct_dim  | Width of a full input in tiles             | uint32_t  | >= block_ct_dim | False    |
 * | Function   | icb          | Input circular buffer identifier           | uint32_t  | 0 to 31         | True     |
 * | Function   | ocb          | Output circular buffer identifier          | uint32_t  | 0 to 31         | True     |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false, false, icb)));  // init must be after configure
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    pack_untilize_dst_init<block_ct_dim, full_ct_dim>(ocb);
}

// clang-format off
/**
 * Performs the untilize operation on a block of tiles. Loops over the provided block size.
 *
 * | Param Type | Name         | Description                                 | Type      | Valid Range     | Required |
 * |------------|--------------|---------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles            | uint32_t  | 1 to 8          | False    |
 * | Template   | full_ct_dim  | Width of a full input in tiles              | uint32_t  | >= block_ct_dim | False    |
 * | Function   | icb          | Input circular buffer identifier            | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim | Height of a single block in tiles           | uint32_t  | >= 1            | True     |
 * | Function   | ocb          | Output circular buffer identifier           | uint32_t  | 0 to 31         | True     |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb) {
    for (uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
            UNPACK(
                (llk_unpack_A<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(icb, c)));
            MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(c)));
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(1 /*num_blocks*/, ocb)));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Performs the pack untilize operation when PACK input is already in DEST register.
 *
 * | Param Type | Name           | Description                                                   | Type      | Valid Range     | Required |
 * |------------|----------------|---------------------------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                              | uint32_t  | 1 to 8          | False    |
 * | Template   | full_ct_dim    | Width of a full input in tiles                                | uint32_t  | >= block_ct_dim | False    |
 * | Template   | diagonal       | Whether to use diagonal packing                               | bool      | true/false      | False    |
 * | Template   | narrow_row     | Whether the provided input is narrow                          | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row                                      | uint32_t  | >= 1            | False    |
 * | Function   | ocb            | Output circular buffer identifier                             | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim   | Height of a single block in tiles                             | uint32_t  | >= 1            | False    |
 * | Function   | block_c_index  | Block column index (used when full_ct_dim > block_ct_dim)     | uint32_t  | >= 0            | False    |
 * | Function   | face_r_dim     | Face height in rows                                           | uint32_t  | 1, 8 or 16      | False    |
 * | Function   | num_faces      | Number of faces                                               | uint32_t  | 1, 2 or 4       | False    |
 */
// clang-format on
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

// clang-format off
/**
 * Uninitializes the pack untilize operation, allowing another operations to be initialized.
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025.
 *
 * | Param Type | Name | Description                        | Type     | Valid Range | Required |
 * |------------|------|------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | Output circular buffer identifier  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void pack_untilize_uninit(uint32_t ocb) {
    PACK((llk_pack_init(ocb)));
    PACK((llk_init_packer_dest_offset_registers<false>()));

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_untilize_uninit(ocb)));
#endif
}

}  // namespace ckernel
