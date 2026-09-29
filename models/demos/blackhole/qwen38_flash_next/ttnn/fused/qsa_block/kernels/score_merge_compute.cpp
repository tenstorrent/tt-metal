// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge compute (16-bit dest, moreh_sum's default): per row chunk the four device chunks accumulated into the
// dest with moreh_sum_nc.cpp's add_tiles(in, zero, acc_to_dest) sequence and packed bf16, then the mask added on the
// SFPU with binary_ng's add_binary_tile<NearestEven>.  Elementwise throughout, so the 2 KB ROW_MAJOR chunks stand in
// for tiles.  CBs: 0 device chunks, 1 zero, 2 mask, 3 sum, 16 out (bf16).  Runtime args: 0 rows, 1 chunks on this core
// (the reader streams the core's chunks one after another, rows within a chunk).

#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "../../kernels/zones.h"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t chunks = get_arg_val<uint32_t>(1);
    constexpr uint32_t CB_IN = 0, CB_ZERO = 1, CB_MASK = 2, CB_SUM = 3, CB_OUT = 16, DEVICES = 4;
    compute_kernel_hw_startup(CB_IN, CB_ZERO, CB_SUM);
    cb_wait_front(CB_ZERO, 1);
    for (uint32_t r = 0; r < rows * chunks;
         ++r) {  // the reader streams the core's chunks one after another, rows within a chunk
        FUSED_ZONE("fz_qs_sm_c_row");
        cb_wait_front(CB_IN, DEVICES);
        cb_reserve_back(CB_SUM, 1);
        reconfig_data_format(CB_IN, CB_ZERO);
        pack_reconfig_data_format(CB_SUM);
        tile_regs_acquire();
        add_init(CB_IN, CB_ZERO, true);
        for (uint32_t d = 0; d < DEVICES; ++d) {
            add_tiles(CB_IN, CB_ZERO, d, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SUM);
        tile_regs_release();
        cb_push_back(CB_SUM, 1);
        cb_pop_front(CB_IN, DEVICES);

        cb_wait_front(CB_SUM, 1);
        cb_wait_front(CB_MASK, 1);
        cb_reserve_back(CB_OUT, 1);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_acquire();
        copy_init(CB_SUM);
        copy_tile(CB_SUM, 0, 0);
        copy_init(CB_MASK);
        copy_tile(CB_MASK, 0, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        cb_push_back(CB_OUT, 1);
        cb_pop_front(CB_SUM, 1);
        cb_pop_front(CB_MASK, 1);
    }
}
