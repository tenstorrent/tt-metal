// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define ELTWISE_OP_TYPE EltwiseBinaryType::ELWADD  // TODO(AP): temporary - refactor

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

#ifdef TRISC_MATH
#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_datacopy_api.h"

void math_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure<DST_ACCUM_MODE>(0, 1);
    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            // Untilize
            llk_math_eltwise_unary_datacopy_init<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(0);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE>(0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }

            llk_math_eltwise_binary_init<ELWADD, NONE>();
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_binary<ELWADD, NONE, DST_ACCUM_MODE, MATH_FIDELITY, EltwiseBinaryReuseDestType::NONE>(0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }
        }
    }
}
#endif

#ifdef TRISC_UNPACK
#include <cstdint>
#include "llk_unpack_common_api.h"
#include "llk_unpack_AB_api.h"
#include "llk_unpack_untilize_api.h"

void unpack_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_unpack_hw_configure<DST_ACCUM_MODE>(0, 1);

    // llk_unpack_untilize_init(0);
    for (uint32_t block = 0U; block < per_core_num_blocks; ++block) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_unpack_untilize_init(0);
            llk_wait_tiles(0, per_core_block_c_tiles);
            llk_unpack_untilize(0, per_core_block_c_tiles);
            llk_unpack_untilize_uninit(0);
            llk_pop_tiles(0, per_core_block_c_tiles);
            llk_pop_tiles(1, per_core_block_c_tiles);

            llk_unpack_AB_init<BroadcastType::NONE>(0, 1);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_wait_tiles(24, 1);
                llk_wait_tiles(1, 1);
                llk_unpack_AB(24, 1, 0, 0);
                llk_pop_tiles(24, 1);
                llk_pop_tiles(1, 1);
            }
        }
    }
}
#endif

#ifdef TRISC_PACK
#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"

void pack_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);
    llk_pack_init();
    llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(16);
    llk_pack_dest_init<DST_ACCUM_MODE, false>();

    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_wait_for_free_tiles<false, false, false>(24, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false, false>(0, 24);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            llk_push_tiles<false, false>(24, per_core_block_c_tiles);

            llk_wait_for_free_tiles<false, false, false>(16, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false, false>(0, 16);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            llk_push_tiles<false, false>(16, per_core_block_c_tiles);
        }
    }
}
#endif

}  // namespace NAMESPACE
