// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// reconfig.cpp over the 2.0 host API: out0 = in1 + in0 + in2 (in2 copied to DEST, then add with acc_to_dest),
// packed to out0 and, after a pack reconfig, to out1. in0/out1 use a different format than in1/in2/out0.
void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t ublock_size_tiles = get_arg(args::ublock_size_tiles);
    constexpr bool explicit_reconfig = get_arg(args::explicit_reconfig) != 0;
    constexpr bool split_src_reconfig = get_arg(args::split_src_reconfig) != 0;
    constexpr bool block_copy = get_arg(args::block_copy) != 0;
    constexpr bool l1_acc = get_arg(args::l1_acc) != 0;

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_in2(dfb::in2);
    DataflowBuffer dfb_out0(dfb::out0);
    DataflowBuffer dfb_out1(dfb::out1);
    const uint32_t in0 = dfb_in0.get_id();
    const uint32_t in1 = dfb_in1.get_id();
    const uint32_t in2 = dfb_in2.get_id();
    const uint32_t out0 = dfb_out0.get_id();
    const uint32_t out1 = dfb_out1.get_id();

    compute_kernel_hw_startup(in0, in1, out0);
    binary_tiles_init<false, EltwiseBinaryType::ELWADD>(in0, in1);
    for (uint32_t block = 0; block < num_tiles; block += ublock_size_tiles) {
        dfb_in0.wait_front(ublock_size_tiles);
        dfb_in1.wait_front(ublock_size_tiles);
        dfb_out0.reserve_back(ublock_size_tiles);
        dfb_out1.reserve_back(ublock_size_tiles);

        tile_regs_acquire();

        // Copy in2 to DEST. The first copy_init targets in0's format; the reconfig + second init switch to in2's.
        copy_init(in0);
        reconfig_data_format_srca(in0, in2);
        copy_init(in2);

        dfb_in2.wait_front(ublock_size_tiles);
        if constexpr (block_copy) {
            copy_block(in2, 0, 0, ublock_size_tiles);
        } else {
            for (uint32_t i = 0; i < ublock_size_tiles; ++i) {
                copy_tile(in2, i, i);
            }
        }
        dfb_in2.pop_front(ublock_size_tiles);

        // SrcA <- in1, SrcB <- in0.
        if constexpr (explicit_reconfig) {
            if constexpr (split_src_reconfig) {
                reconfig_data_format_srca(in0, in1);
                reconfig_data_format_srcb(in1, in0);
            } else {
                reconfig_data_format(in0, in1, in1, in0);
            }
        } else {
            if constexpr (split_src_reconfig) {
                reconfig_data_format_srca(in1);
                reconfig_data_format_srcb(in0);
            } else {
                reconfig_data_format(in1, in0);
            }
        }

        add_init(in1, in0, true /* acc_to_dest */);
        for (uint32_t i = 0; i < ublock_size_tiles; ++i) {
            add_tiles(in1, in0, i, i, i);
        }

        tile_regs_commit();
        tile_regs_wait();

        if constexpr (l1_acc) {
            // Seed the out0 entry with a plain pack, then pack the same tiles again with L1 accumulation on, so
            // out0 = 2 * result regardless of what the entry held before.
            pack_reconfig_l1_acc(false);
            for (uint32_t i = 0; i < ublock_size_tiles; ++i) {
                pack_tile<true /* out_of_order_output */>(i, out0, i);
            }
            pack_reconfig_l1_acc(true);
            for (uint32_t i = 0; i < ublock_size_tiles; ++i) {
                pack_tile<true /* out_of_order_output */>(i, out0, i);
            }
        } else {
            pack_block(0, out0, ublock_size_tiles);
        }

        if constexpr (explicit_reconfig) {
            pack_reconfig_data_format(out0, out1);
        } else {
            pack_reconfig_data_format(out1);
        }
        pack_init(out1);
        pack_reconfig_l1_acc(false);
        pack_block(0, out1, ublock_size_tiles);

        // Restore out0 for the next block.
        pack_reconfig_data_format(out1, out0);
        pack_init(out0);
        tile_regs_release();

        dfb_in0.pop_front(ublock_size_tiles);
        dfb_in1.pop_front(ublock_size_tiles);
        dfb_out0.push_back(ublock_size_tiles);
        dfb_out1.push_back(ublock_size_tiles);
    }
}
