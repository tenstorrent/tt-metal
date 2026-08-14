// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Alternating-format copy, LLK 1.0 style — the inner-loop shape of the sort /
// SDPA kernels: two streams of different data formats (values: Float16_b,
// indices: UInt16) alternate through the same SrcA -> DST -> pack path. Every
// half-iteration swaps the SrcA and pack formats, which in LLK 1.0 costs a
// reconfig_data_format_srca + copy_tile_to_dst_init_short pair on
// UNPACK/MATH and a pack_reconfig_data_format on PACK — exactly the pattern
// at sort_single_row_multi_core.cpp:136-145 / 186-192.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_iters = get_arg(args::num_iters);
    constexpr uint32_t tiles_per_block = 2;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out0(dfb::out0);
    DataflowBuffer dfb_out1(dfb::out1);

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out0);

    for (uint32_t it = 0; it < num_iters; ++it) {
        // Stream A (Float16_b values).
        reconfig_data_format_srca(dfb::in0);
        copy_tile_to_dst_init_short(dfb::in0);
        dfb_in0.wait_front(tiles_per_block);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            copy_tile(dfb::in0, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        dfb_out0.reserve_back(tiles_per_block);
        pack_reconfig_data_format(dfb::out0);
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            pack_tile(i, dfb::out0);
        }
        tile_regs_release();
        dfb_in0.pop_front(tiles_per_block);
        dfb_out0.push_back(tiles_per_block);

        // Stream B (UInt16 indices).
        reconfig_data_format_srca(dfb::in1);
        copy_tile_to_dst_init_short(dfb::in1);
        dfb_in1.wait_front(tiles_per_block);
        tile_regs_acquire();
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            copy_tile(dfb::in1, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        dfb_out1.reserve_back(tiles_per_block);
        pack_reconfig_data_format(dfb::out1);
        for (uint32_t i = 0; i < tiles_per_block; ++i) {
            pack_tile(i, dfb::out1);
        }
        tile_regs_release();
        dfb_in1.pop_front(tiles_per_block);
        dfb_out1.push_back(tiles_per_block);
    }
}
