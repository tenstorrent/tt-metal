// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_untilize_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Performs the necessary hardware and software initialization for the untilize operation. This function should be
 * called before performing untilize operation in the compute kernel. If the data format or properties of the input
 * operand differ from those previously configured, ensure that the appropriate reconfiguration functions are called
 * before this initialization.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                                               | Type     | Valid Range | Required |
 * |------------|------|-----------------------------------------------------------|----------|-------------|----------|
 * | Function   | icb  | The identifier of the circular buffer (CB) for input data | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void untilize_init(uint32_t icb) {
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    UNPACK((llk_unpack_untilize_init(icb)));
}

// clang-format off
/**
 * Copies a block of tiles from the DEST register buffer to the specified circular buffer (CB) for the untilize
 * operation. This function is used to transfer multiple tiles at once, with the number of tiles determined by the
 * template parameter `block_ct_dim`. The DEST register buffer must be in the acquired state before calling this function. The CB
 * ID provided must correspond to the buffer where the untilized data will be stored. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                          | Type     | Valid Range               | Required |
 * |------------|--------------|------------------------------------------------------|----------|---------------------------|----------|
 * | Template   | block_ct_dim | The number of tiles stored in DEST at one moment     | uint32_t | 1 to max (see comment)    | False    |
 * | Function   | icb          | The identifier of the circular buffer (CB) for input | uint32_t | 0 to 31                   | True     |
 * | Function   | full_ct_dim  | Width of a full input in tiles                       | uint32_t | Divisible by block_ct_dim | True     |
 * | Function   | ocb          | The identifier of the circular buffer (CB) for output| uint32_t | 0 to 31                   | True     |
 */
// clang-format on
template <uint32_t block_ct_dim = 1>
ALWI void untilize_block(uint32_t icb, uint32_t full_ct_dim, uint32_t ocb) {
    UNPACK((llk_unpack_untilize(icb, full_ct_dim)));

    for (uint32_t t = 0; t < full_ct_dim / block_ct_dim; t++) {
        MATH((llk_math_wait_for_dest_available()));

        // Datacopy
        for (uint32_t reg_id = 0; reg_id < block_ct_dim; reg_id++) {
            MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(reg_id)));
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));

        // Datacopy
        for (uint32_t reg_id = 0; reg_id < block_ct_dim; reg_id++) {
            PACK((llk_pack<DST_ACCUM_MODE, false, false>(reg_id, ocb)));
        }

        // Release dest
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Restores hardware and internal state after the untilize operation is complete. This function should be called after
 * untilize_block operation is finished. The circular buffer (CB) ID provided must correspond to the buffer that was
 * used for the untilize operation and it's initialization.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                                               | Type     | Valid Range | Required |
 * |------------|------|-----------------------------------------------------------|----------|-------------|----------|
 * | Function   | icb  | The identifier of the circular buffer (CB) for input data | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void untilize_uninit(uint32_t icb) { UNPACK((llk_unpack_untilize_uninit(icb))); }

}  // namespace ckernel
