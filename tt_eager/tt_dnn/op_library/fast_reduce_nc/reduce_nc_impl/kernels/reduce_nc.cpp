// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
void MAIN {
    // compile-time args
    constexpr uint32_t num_output_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_input_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t input_granularity = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    constexpr uint32_t num_input_tiles_iter = num_input_tiles / input_granularity;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    cb_wait_front(cb_in1, onetile);

    DPRINT_PACK(DPRINT << "Checkpoint 1" << ENDL());
    DPRINT_PACK(DPRINT << "num_output_tiles: " << num_output_tiles << ENDL());
    DPRINT_PACK(DPRINT << "num_input_tiles: " << num_input_tiles << ENDL());
    DPRINT_PACK(DPRINT << "num_input_tiles_iter: " << num_input_tiles_iter << ENDL());

    for (uint32_t i = 0; i < num_output_tiles; i++) {

        add_tiles_init(cb_in1, cb_in1);
        #if defined FP32_DEST_ACC_EN
            unpack_reconfig_data_format(cb_in1, cb_in1);
            pack_reconfig_data_format(cb_intermed0);
        #endif
        tile_regs_acquire();
        add_tiles(cb_in1, cb_in1, first_tile, first_tile, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_intermed0);
        tile_regs_release();
        cb_push_back(cb_intermed0, onetile);

        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            bool last_out = (j == num_input_tiles_iter - 1);

            cb_wait_front(cb_in0, input_granularity);
            // DPRINT_PACK(DPRINT << "Iter: "<< i << "," << j << ". Checkpoint 2" << ENDL());

            add_tiles_init(cb_in0, cb_intermed0);
            #if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_intermed0);
                unpack_reconfig_data_format(cb_in0, cb_intermed0);
            #endif
            for (uint32_t k = 0; k < input_granularity; k++) {
                cb_wait_front(cb_intermed0, onetile);
                tile_regs_acquire();
                add_tiles(cb_in0, cb_intermed0, k, first_tile, dst0);
                tile_regs_commit();
                if (k < input_granularity - 1) {
                    cb_pop_front(cb_intermed0, onetile);
                    tile_regs_wait();
                    pack_tile(dst0, cb_intermed0);
                    tile_regs_release();
                    cb_push_back(cb_intermed0, onetile);
                }
            }

            // DPRINT << "Iter: "<< i << "," << j << ". Checkpoint 3" << ENDL();

            cb_pop_front(cb_in0, input_granularity);
            cb_pop_front(cb_intermed0, onetile);

            uint32_t cb_out = (last_out) ? (cb_out0) : (cb_intermed0);
            cb_reserve_back(cb_out, onetile);
            #if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_out);
            #endif
            tile_regs_wait();
            pack_tile(dst0, cb_out);
            tile_regs_release();
            // DPRINT << "Iter: "<< i << "," << j << ". Checkpoint 4" << ENDL();

            // if (last_out) {
            //     DPRINT_PACK(DPRINT << "Iter: "<< i << "," << j << ". last out" << ENDL());
            // }

            cb_push_back(cb_out, onetile);

            // DPRINT_PACK(DPRINT << "Iter: "<< i << "," << j << ". Checkpoint 5" << ENDL());
        }
    }
}
}  // namespace NAMESPACE
