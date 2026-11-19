// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"
void load(uint32_t cb,uint32_t dst) {
    reconfig_data_format_srca(cb);copy_init(cb);copy_tile(cb,0,dst);
}
void kernel_main() {
    compute_kernel_hw_startup(0,7,16);
    for(uint32_t cb=0;cb<8;++cb) { CircularBuffer input(cb);input.wait_front(1); }
    CircularBuffer output(16);output.reserve_back(1);
    tile_regs_acquire();
    load(0,0);load(7,1);
    sfpu_bcast_row_init();sfpu_mul_bcast_row(0,1);
    for(uint32_t j=0;j<3;++j) {
        load(1+j,2);load(4+j,3);
        sfpu_bcast_row_init();sfpu_mul_bcast_row(2,3);
        add_binary_tile_init();add_binary_tile(0,2,0);
    }
    silu_tile_init();silu_tile(0);
    tile_regs_commit();tile_regs_wait();
    pack_reconfig_data_format(16);pack_tile(0,16,0);
    tile_regs_release();output.push_back(1);
    // Inputs deliberately stay published for history writes by the writer.
}
