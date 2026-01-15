// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_datacopy_api.h"

namespace NAMESPACE {

void math_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure<DST_ACCUM_MODE>(0, 1);
    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            // Untilize
            llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }

            llk_math_eltwise_binary_init<ELWADD, NONE>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, false>(0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }
        }
    }
}
}  // namespace NAMESPACE
