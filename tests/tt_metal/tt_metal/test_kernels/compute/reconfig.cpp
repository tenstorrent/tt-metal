// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"
#include <cstdint>
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/unpack.h"

#define START_IN_TILE_ID              (0)
#define START_DST_TILE_ID             (0)

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t ublock_size_tiles = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CB::c_in0; // Bfp8_b
    constexpr auto cb_in1 = tt::CB::c_in1; // Bfp16_b
    constexpr auto cb_in2 = tt::CB::c_in2; // Bfp16_b
    constexpr auto cb_out0 = tt::CB::c_out0; // Bfp16_b
    constexpr auto cb_out1 = tt::CB::c_out1; // Bfp8_b


    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init_nof();
    for (uint32_t block = 0; block < num_tiles; ++block) {

        cb_wait_front(cb_in0, ublock_size_tiles);
        cb_wait_front(cb_in1, ublock_size_tiles);
        cb_reserve_back(cb_out0, ublock_size_tiles);
        cb_reserve_back(cb_out1, ublock_size_tiles);

        acquire_dst(tt::DstMode::Half);

        // ------------------------- Copy to DEST -----------------------------

        // Tests both inits, 1st one inits UNPACK for Bfp8_b
        // data inside CB_0, 2nd one inits it to Bfp16_b
        // which is inside CB_2
        copy_tile_init();
        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in2);

        cb_wait_front(cb_in2, ublock_size_tiles);
        copy_block_matmul_partials(cb_in2, START_IN_TILE_ID, START_DST_TILE_ID, ublock_size_tiles);
        cb_pop_front(cb_in2, ublock_size_tiles);

        // -------------------- Addition with acc -----------------------------

        // Init like CB_0 is in A and CB_1 is in B
        add_tiles_init(cb_in0, cb_in1, true);

        // Reconfigure UNPACK for correct source formats, tests reconfig calls
#if (EXPLICIT_RECONFIG == 1)
#if (SPLIT_SRC_RECONFIG == 1)
        // Indices for old_operand, new_operand
        unpack_reconfig_data_format_srca(cb_in0, cb_in1);
        unpack_reconfig_data_format_srcb(cb_in1, cb_in0);
#elif (SPLIT_SRC_RECONFIG == 0)
        // Indices for old_A, new_A, old_B, new_B
        unpack_reconfig_data_format(cb_in0, cb_in1, cb_in1, cb_in0);
#endif // SPLIT_SRC_RECONFIG
#elif (EXPLICIT_RECONFIG == 0)
#if (SPLIT_SRC_RECONFIG == 1)
        // Indices for new_operand
        unpack_reconfig_data_format_srca(cb_in1);
        unpack_reconfig_data_format_srcb(cb_in0);
#elif (SPLIT_SRC_RECONFIG == 0)
        // Indices for new_A, new_B
        unpack_reconfig_data_format(cb_in1, cb_in0);
#endif // SPLIT_SRC_RECONFIG
#endif // EXPLICIT_RECONFIG

        for (uint32_t i = 0; i < ublock_size_tiles; ++i) {
            add_tiles(cb_in1, cb_in0, i, i, i);
        }

        // ----------------------- Pack to 2 outs -----------------------------

        // Reconfig for L1 accumulation with old calc values
#if (L1_ACC == 1)
        pack_reconfig_l1_acc(true);
#endif
        // Configured already for CB_16, Bfp16_b
        matmul_pack_tile(START_DST_TILE_ID, cb_out0, ublock_size_tiles);
        // Reconfig for CB_17, Bfp8_b, then pack to CB_17
#if (EXPLICIT_RECONFIG == 1)
        // Indices for old_output, new_output
        pack_reconfig_data_format(cb_out0, cb_out1);
#elif (EXPLICIT_RECONFIG == 0)
        // Indices for new_output
        pack_reconfig_data_format(cb_out1);
#endif
        // Not testing for L1 accumulation
        pack_reconfig_l1_acc(false);

        matmul_pack_tile(START_DST_TILE_ID, cb_out1, ublock_size_tiles);
        release_dst(tt::DstMode::Half);

        cb_pop_front(cb_in0, ublock_size_tiles);
        cb_pop_front(cb_in1, ublock_size_tiles);
        cb_push_back(cb_out0, ublock_size_tiles);
        cb_push_back(cb_out1, ublock_size_tiles);
    }
}
}  // namespace NAMESPACE
