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
#include "api/compute/eltwise_unary/softplus.h"
void load(uint32_t cb,uint32_t dst) {
    reconfig_data_format_srca(cb);copy_init(cb);copy_tile(cb,0,dst);
}
void gate_prepare() {
    for(uint32_t cb=8;cb<12;++cb)CircularBuffer(cb).wait_front(1);
    CircularBuffer beta(18);beta.reserve_back(1);
    tile_regs_acquire();load(8,0);sigmoid_tile_init();sigmoid_tile(0);
    tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(18);pack_tile(0,18,0);
    tile_regs_release();beta.push_back(1);

    CircularBuffer sum(12);sum.reserve_back(1);
    tile_regs_acquire();load(9,0);load(10,1);sfpu_bcast_row_init();sfpu_add_bcast_row(0,1);
    tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(12);pack_tile(0,12,0);
    tile_regs_release();sum.push_back(1);sum.wait_front(1);

    CircularBuffer soft(13);soft.reserve_back(1);
    tile_regs_acquire();load(12,0);softplus_tile_init();softplus_tile(0,0x3f800000u,0x3f800000u,0x41a00000u);
    tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(13);pack_tile(0,13,0);
    tile_regs_release();soft.push_back(1);sum.pop_front(1);soft.wait_front(1);

    CircularBuffer g(17);g.reserve_back(1);
    tile_regs_acquire();load(13,0);load(11,1);sfpu_bcast_row_init();sfpu_mul_bcast_row(0,1);
    tile_regs_commit();tile_regs_wait();pack_reconfig_data_format(17);pack_tile(0,17,0);
    tile_regs_release();g.push_back(1);soft.pop_front(1);
    for(uint32_t cb=8;cb<12;++cb)CircularBuffer(cb).pop_front(1);
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
    if(get_arg_val<uint32_t>(0)==0)gate_prepare();
    // Inputs deliberately stay published for history writes by the writer.
}
