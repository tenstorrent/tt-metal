// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// score_merge compute (16-bit dest, moreh_sum's default): per row chunk the four device chunks accumulated into the
// dest with moreh_sum_nc.cpp's add_tiles(in, zero, acc_to_dest) sequence and packed bf16, then the mask added on the
// SFPU with binary_ng's add_binary_tile<NearestEven>.  Elementwise throughout, so the 2 KB ROW_MAJOR chunks stand in
// for tiles.  CBs: 0 device chunks, 1 zero, 2 mask, 3 sum, 16 out (bf16).  Runtime args: 0 rows, 1 chunks on this core.

#include "api/compute/common.h"
#include "api/dataflow/circular_buffer.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t chunks = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_in = 0, cb_zero = 1, cb_mask = 2, cb_sum = 3, cb_out = 16, devices = 4;
    compute_kernel_hw_startup(cb_in, cb_zero, cb_sum);
    CircularBuffer(cb_zero).wait_front(1);
    for (uint32_t r = 0; r < rows * chunks;
         ++r) {  // the reader streams the core's chunks one after another, rows within a chunk
        CircularBuffer(cb_in).wait_front(devices);
        CircularBuffer(cb_sum).reserve_back(1);
        reconfig_data_format(cb_in, cb_zero);
        pack_reconfig_data_format(cb_sum);
        tile_regs_acquire();
        add_init(cb_in, cb_zero, true);
        for (uint32_t d = 0; d < devices; ++d) {
            add_tiles(cb_in, cb_zero, d, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_sum);
        tile_regs_release();
        CircularBuffer(cb_sum).push_back(1);
        CircularBuffer(cb_in).pop_front(devices);

        CircularBuffer(cb_sum).wait_front(1);
        CircularBuffer(cb_mask).wait_front(1);
        CircularBuffer(cb_out).reserve_back(1);
        pack_reconfig_data_format(cb_out);
        tile_regs_acquire();
        copy_init(cb_sum);
        copy_tile(cb_sum, 0, 0);
        copy_init(cb_mask);
        copy_tile(cb_mask, 0, 1);
        add_binary_tile_init();
        add_binary_tile<ckernel::DstRoundingMode::NearestEven>(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
        CircularBuffer(cb_out).push_back(1);
        CircularBuffer(cb_sum).pop_front(1);
        CircularBuffer(cb_mask).pop_front(1);
    }
}
