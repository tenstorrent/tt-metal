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

/**
 * Init function for untilize operations, to be used at the beginning of the kernel.
 */
ALWI void untilize_init(uint32_t icb, uint32_t ocb) {
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(icb, icb)));

    PACK((llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<false, DST_ACCUM_MODE>()));

    UNPACK((llk_unpack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE>(icb)));
    UNPACK((llk_unpack_untilize_init(icb)));  // init must be after configure
}

/**
 * Short init function to initialize untilize op, after a full init is already performed.
 */
ALWI void untilize_init_short(uint32_t icb) {
    MATH((llk_math_eltwise_unary_datacopy_init<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(
        false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
    UNPACK((llk_unpack_untilize_init(icb)));
}

/**
 * Perform the untilize operation on a block of tiles. This simply loops over the provided block size.
 */
template <int N = 1>
ALWI void untilize_block(uint32_t icb, uint32_t block, uint32_t ocb) {
    UNPACK((llk_unpack_untilize(icb, block)));

    for (uint32_t t = 0; t < block / N; t++) {
        MATH((llk_math_wait_for_dest_available()));

        // Datacopy
        for (int reg_id = 0; reg_id < N; reg_id++) {
            MATH((llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, DST_ACCUM_MODE>(reg_id)));
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));

        PACK((llk_packer_wait_for_math_done()));

        // Datacopy
        for (int reg_id = 0; reg_id < N; reg_id++) {
            PACK((llk_pack<false, false, DST_ACCUM_MODE>(reg_id, ocb)));
        }

        // Release dest
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

/**
 * Uninitialize untilize operation, to allow initializing another operation.
 */
ALWI void untilize_uninit(uint32_t icb) { UNPACK((llk_unpack_untilize_uninit(icb))); }

}  // namespace ckernel
