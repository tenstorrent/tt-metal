// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// main_tail, staging core compute (fp32 dest): per lane the patched staging tiles times the ones tile on the SFPU
// (matches the one-hot select: bf16 RNE store, zero rule), kept for the TILE write-back and untilized into 32
// row-major rows for the cache write.  Runtime arg 0: this core's lane count.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "main_tail_cbs.h"

using namespace main_tail;

void kernel_main() {
    const uint32_t lane_count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_stg, cb_ones, cb_stgc);
    CircularBuffer(cb_ones).wait_front(1);
    for (uint32_t i = 0; i < lane_count; ++i) {
        CircularBuffer(cb_stg).wait_front(pack_tiles);
        CircularBuffer(cb_stgc).reserve_back(pack_tiles);
        CircularBuffer(cb_stgw).reserve_back(pack_tiles);
        reconfig_data_format(cb_stg, cb_ones);
        pack_reconfig_data_format(cb_stgc);
        mul_binary_tile_init();
        for (uint32_t t = 0; t < pack_tiles; ++t) {
            tile_regs_acquire();
            copy_init(cb_stg);
            copy_tile(cb_stg, t, 0);
            copy_init(cb_ones);
            copy_tile(cb_ones, 0, 1);
            mul_binary_tile<false>(0, 1, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_stgc);
            tile_regs_release();
        }
        CircularBuffer(cb_stgc).push_back(pack_tiles);
        CircularBuffer(cb_stgw).push_back(pack_tiles);
        CircularBuffer(cb_stg).pop_front(pack_tiles);

        reconfig_data_format(cb_stgc, cb_stgc);
        pack_reconfig_data_format(cb_rm);
        compute_kernel_lib::untilize<
            pack_tiles,
            cb_stgc,
            cb_rm,
            compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    }
}
