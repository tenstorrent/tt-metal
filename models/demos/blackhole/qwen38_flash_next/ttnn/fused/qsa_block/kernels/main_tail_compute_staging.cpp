// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core compute (fp32 dest): per lane the patched staging tiles times the ones tile on the SFPU
// (the chain's one-hot select arithmetic: bf16 RNE store, zero rule), kept for the writer (TILE write-back) and
// untilized into 32 row-major rows for the cache write (the chain's to_layout).  Runtime arg 0: this core's lane
// count (one staging core per lane).

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_STG, CB_ONES, CB_STGC);
    cb_wait_front(CB_ONES, 1);
    for (uint32_t i = 0; i < lane_count; ++i) {
        cb_wait_front(CB_STG, PACK_TILES);
        cb_reserve_back(CB_STGC, PACK_TILES);
        cb_reserve_back(CB_STGW, PACK_TILES);
        reconfig_data_format(CB_STG, CB_ONES);
        pack_reconfig_data_format(CB_STGC);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < PACK_TILES; ++t) {
            tile_regs_acquire();
            copy_init(CB_STG);
            copy_tile(CB_STG, t, 0);
            copy_init(CB_ONES);
            copy_tile(CB_ONES, 0, 1);
            mul_binary_tile<false>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, CB_STGC);
            tile_regs_release();
        }
        cb_push_back(CB_STGC, PACK_TILES);
        cb_push_back(CB_STGW, PACK_TILES);
        cb_pop_front(CB_STG, PACK_TILES);

        reconfig_data_format(CB_STGC, CB_STGC);
        pack_reconfig_data_format(CB_RM);
        compute_kernel_lib::untilize<
            PACK_TILES,
            CB_STGC,
            CB_RM,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }
}
