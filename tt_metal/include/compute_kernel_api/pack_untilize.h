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
 * configure UNPACK and MATH threads for transferring data from circular buffers to DEST register (this is done with
 * `pack_untilize_init` function). Matching pack untilize operation for this initialization function is
 * `pack_untilize_dest`. In order for this untilization to be performed correctly, some other function must
 * place the tiles in the DEST register, e.g. `reduce_tile`, `copy_tile`, etc. This initialization function
 * should, therefore, be called right after the op-specific initialization function (`reduce_init`, `copy_init`, etc.).
 *
 * Since pack untilize works on a block of tiles, the user should specify the width of a single block (block_ct_dim),
 * and the width of the full block (`full_ct_dim`). It is not needed to provide the height of the block during the
 * initialization, since `pack_untilize_block` will loop over the height of the block. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * NOTE: This function allows the user to specify `face_r_dim` and `num_faces` through function parameters. Setting these
 * parameters results in an expensive MMIO write and cannot be avoided currently.
 * This should be addressed more systematically within the issue tt-metal#22820, since these two values can be inferred
 * from the circular buffer description, the same way as it is done in `llk_pack_hw_configure_disaggregated`. This
 * would remove the need for `llk_pack_untilize_hw_configure_disaggregated` altogether and we would pay the price
 * of the MMIO write only once, in `compute_kernel_hw_startup`.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                              | Type      | Valid Range     | Required |
 * |------------|--------------|------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles         | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim  | Width of a full input in tiles           | uint32_t  | Divisible by block_ct_dim | False    |
 * | Template   | narrow_row   |  Whether the provided input is narrow    | bool      | true/false      | False    |
 * | Template   | row_num_datums | Number of datums per row               | uint32_t  | >= 1            | False    |
 * | Function   | ocb          | Output circular buffer identifier        | uint32_t  | 0 to 31         | True     |
 * | Function   | face_r_dim   | Face height in rows                      | uint32_t  | 1, 8 or 16      | False (default = 16) |
 * | Function   | num_faces    | Number of faces                          | uint32_t  | 1, 2 or 4       | False (default = 4) |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM>
ALWI void pack_untilize_dest_init(uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4) {
#ifdef ARCH_BLACKHOLE
    // Needed for setting swizzle_32b:
    MATH((llk_math_reconfig_remap(true)));
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
 * register. Internally, this function calls `pack_untilize_dest_init` to initialize the PACK thread.
 *
 * Since pack untilize works on a block of tiles, the user should specify the width of a single block (block_ct_dim),
 * and the width of the full block (`full_ct_dim`). It is not needed to provide the height of the block during the
 * initialization, since `pack_untilize_block` will loop over the height of the block. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * NOTE: This function uses default `face_r_dim` and `num_faces` values (16 and 4, respectively). Setting these
 * parameters results in an expensive MMIO write and cannot be avoided currently.
 * This should be addressed more systematically within the issue tt-metal#22820, since these two values can be inferred
 * from the circular buffer description, the same way as it is done in `llk_pack_hw_configure_disaggregated`. This
 * would remove the need for `llk_pack_untilize_hw_configure_disaggregated` altogether and we would pay the price
 * of the MMIO write only once, in `compute_kernel_hw_startup`.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                | Type      | Valid Range     | Required |
 * |------------|--------------|--------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles           | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim  | Width of a full input in tiles             | uint32_t  | Divisible by block_ct_dim | False    |
 * | Function   | icb          | Input circular buffer identifier           | uint32_t  | 0 to 31         | True     |
 * | Function   | ocb          | Output circular buffer identifier          | uint32_t  | 0 to 31         | True     |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
        false, false, icb)));  // init must be after configure
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(ocb);
}

// clang-format off
/**
 * Performs the untilize operation on a block of tiles. Loops over the provided block size. The block is characterized
 * by its width in tiles (block_ct_dim) and height in tiles (block_rt_dim). The width of the block has to be the same
 * as the one provided during the initialization of the pack untilize operation (`pack_untilize_init`). It is not
 * needed to provide the height of the block during the initialization, since `pack_untilize_block` will loop over the
 * height of the block. Note that the maximum size of the block is limited by the size of the DEST and synchronization
 * mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                 | Type      | Valid Range     | Required |
 * |------------|--------------|---------------------------------------------|-----------|-----------------|----------|
 * | Template   | block_ct_dim | Width of a single block in tiles            | uint32_t  | 1 to max (see note) | False (default = 8) |
 * | Template   | full_ct_dim  | Width of a full input in tiles              | uint32_t  | Divisible by block_ct_dim | False    |
 * | Function   | icb          | Input circular buffer identifier            | uint32_t  | 0 to 31         | True     |
 * | Function   | block_rt_dim | Height of a single block in tiles           | uint32_t  | >= 1            | True     |
 * | Function   | ocb          | Output circular buffer identifier           | uint32_t  | 0 to 31         | True     |
 * | Function   | block_c_index | Index of the currently processed block     | uint32_t  | >= 0            | False    |
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

// clang-format off
/**
 * Performs the pack untilize operation when PACK input is already in DEST register. In order to properly initialize the operation,
 * a call to `pack_untilize_dest_init` must be made before this function. The width of the block has to be the same
 * as the one provided during the initialization of the pack untilize operation (`pack_untilize_dest_init`). In order for this
 * untilization to be performed correctly, some other function must place the tiles in the DEST register, e.g. `reduce_tile`,
 * `copy_tile`, etc. Similarly as `pack_untilize_block`, this function operates on a whole block that needs to be untilized.
 * Note that the maximum size of the block is limited by the size of the DEST
 * and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name               | Description                                                                  | Type      | Valid Range                             | Required            |
 * |------------|--------------------|------------------------------------------------------------------------------|-----------|-----------------------------------------|---------------------|
 * | Template   | block_ct_dim       | Width of a single block in tiles                                             | uint32_t  | 1 to max (see note)                     | False (default = 8) |
 * | Template   | full_ct_dim        | Width of a full input in tiles                                               | uint32_t  | Divisible by block_ct_dim               | False               |
 * | Template   | diagonal           | Whether to use diagonal packing                                              | bool      | true/false                              | False               |
 * | Template   | narrow_row         | Whether the provided input is narrow                                         | bool      | true/false                              | False               |
 * | Template   | row_num_datums     | Number of datums per row                                                     | uint32_t  | >= 1                                    | False               |
 * | Template   | tile_dst_ct_offset | Compile time offset for the index of the tile in the dest from which to pack | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0)   |
 * | Function   | ocb                | Output circular buffer identifier                                            | uint32_t  | 0 to 31                                 | True                |
 * | Function   | block_rt_dim       | Height of a single block in tiles                                            | uint32_t  | >= 1                                    | False (default=1)   |
 * | Function   | block_c_index      | Block column index (used when full_ct_dim > block_ct_dim)                    | uint32_t  | >= 0                                    | False (default=0)   |
 * | Function   | face_r_dim         | Face height in rows                                                          | uint32_t  | 1, 8 or 16                              | False (default=16)  |
 * | Function   | num_faces          | Number of faces                                                              | uint32_t  | 1, 2 or 4                               | False (default=4)   |
 * | Function   | tile_dst_offset    | Runtime offset for the index of the tile in the dest from which to pack      | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0)   |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    uint32_t tile_dst_ct_offset = 0>
ALWI void pack_untilize_dest(
    uint32_t ocb,
    uint32_t block_rt_dim = 1,
    uint32_t block_c_index = 0 /* used when full_ct_dim > block_ct_dim*/,
    uint32_t face_r_dim = 16,
    uint32_t num_faces = 4,
    uint32_t tile_dst_rt_offset = 0) {
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset>(
        block_rt_dim, ocb, face_r_dim, num_faces, block_c_index, tile_dst_rt_offset)));
}

// clang-format off
/**
 * Uninitializes the pack untilize operation, allowing another operations to be initialized. Needs to be
 * called after the last call to `pack_untilize_dest` or `pack_untilize_block`, before initializing
 * another operation.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                        | Type     | Valid Range | Required |
 * |------------|------|------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | Output circular buffer identifier  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void pack_untilize_uninit(uint32_t ocb) {
    PACK((llk_init_packer_dest_offset_registers<false>()));
    PACK((llk_pack_init(ocb)));

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_untilize_uninit(ocb)));
#endif
}

}  // namespace ckernel
