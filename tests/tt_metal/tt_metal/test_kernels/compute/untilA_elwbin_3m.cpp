// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define ELTWISE_OP_TYPE EltwiseBinaryType::ELWADD  // TODO(AP): temporary - refactor

#include "api/compute/eltwise_unary/sfpu_split_includes.h"

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

// CB ids used across all three TRISC sections.
//   cb0      : tilized input A (consumed by untilize)
//   cb1      : input B for the eltwise binary
//   cb24     : untilized A (produced by pack from untilize, consumed by binary unpack)
//   cb16     : final eltwise-binary output
constexpr uint32_t cb0 = tt::CBIndex::c_0;
constexpr uint32_t cb1 = tt::CBIndex::c_1;
constexpr uint32_t cb24 = tt::CBIndex::c_24;
constexpr uint32_t cb16 = tt::CBIndex::c_16;

constexpr uint32_t onetile = 1;

#ifdef TRISC_MATH
#include <cstdint>
#include "llk_math_common.h"
#include "llk_math_binary_api.h"
#include "llk_math_unary_datacopy_api.h"

void core_agnostic_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_math_pack_sync_init<DST_ACCUM_MODE>();
    llk_math_hw_configure<DST_ACCUM_MODE>(cb0, cb1);
    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            // Untilize
            llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(cb0);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(
                    0 /*dst_index*/, cb0);
                llk_math_dest_section_done<DST_ACCUM_MODE>();
            }

            llk_math_eltwise_binary_init<EltwiseBinaryType::ELWADD, BroadcastType::NONE, MathFidelity::LoFi>(cb24, cb1);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_math_wait_for_dest_available();
                llk_math_eltwise_binary<
                    EltwiseBinaryType::ELWADD,
                    BroadcastType::NONE,
                    DST_ACCUM_MODE,
                    MATH_FIDELITY,
                    EltwiseBinaryReuseDestType::NONE>(0 /*dst_index*/);
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

void core_agnostic_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);

    llk_unpack_hw_configure<DST_ACCUM_MODE>(cb0, cb1);

    for (uint32_t block = 0U; block < per_core_num_blocks; ++block) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            llk_unpack_untilize_init(cb0);
            cb_wait_front(cb0, per_core_block_c_tiles);
            llk_unpack_untilize(cb0, per_core_block_c_tiles);
#ifdef ARCH_BLACKHOLE
            llk_unpack_untilize_uninit(cb0);
#else
            llk_unpack_untilize_uninit();
#endif
            cb_pop_front(cb0, per_core_block_c_tiles);
            cb_pop_front(cb1, per_core_block_c_tiles);

            llk_unpack_AB_init<BroadcastType::NONE>(cb24, cb1);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                cb_wait_front(cb24, onetile);
                cb_wait_front(cb1, onetile);
                llk_unpack_AB(cb24, cb1, 0 /*11_tile_index_a*/, 0 /*l1_tile_index_b*/);
                cb_pop_front(cb24, onetile);
                cb_pop_front(cb1, onetile);
            }
        }
    }
}
#endif

#ifdef TRISC_PACK
#include <cstdint>
#include "llk_pack_common.h"
#include "llk_pack.h"

void core_agnostic_main() {
    uint32_t per_core_num_blocks = get_compile_time_arg_val(0);
    uint32_t per_core_block_r_tiles = get_compile_time_arg_val(1);
    uint32_t per_core_block_c_tiles = get_compile_time_arg_val(2);
    llk_pack_init();
    llk_pack_hw_configure<DST_ACCUM_MODE>(cb16);
    llk_pack_dest_init<DST_ACCUM_MODE, PackMode::Default>();

    for (uint32_t block = 0; block < per_core_num_blocks; block++) {
        for (uint32_t r = 0; r < per_core_block_r_tiles; r++) {
            cb_reserve_back(cb24, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false /*out_of_order_output*/, PackMode::Default>(0 /*dst_tile_index*/, cb24);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            cb_push_back(cb24, per_core_block_c_tiles);

            cb_reserve_back(cb16, per_core_block_c_tiles);
            for (uint32_t c = 0; c < per_core_block_c_tiles; c++) {
                llk_packer_wait_for_math_done();
                llk_pack<DST_ACCUM_MODE, false /*out_of_order_output*/, PackMode::Default>(0 /*dst_tile_index*/, cb16);
                llk_pack_dest_section_done<DST_ACCUM_MODE>();
            }
            cb_push_back(cb16, per_core_block_c_tiles);
        }
    }
}
#endif

void kernel_main() {
#if defined(TRISC_MATH) || defined(TRISC_UNPACK) || defined(TRISC_PACK)
    core_agnostic_main();
#endif
}
