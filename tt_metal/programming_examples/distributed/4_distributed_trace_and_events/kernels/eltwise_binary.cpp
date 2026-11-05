// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_binary.h"

#include <cstdint>

#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    uint32_t per_core_block_size = get_arg(args::per_core_block_size);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    // full_init=true: compute_kernel_hw_startup does not run llk_unpack_AB_init, so a math-only
    // (binary_tiles_init<false>) init is no longer sufficient on its own; always do the full init.
    binary_tiles_init<true, ELTWISE_OP_TYPE>(dfb::in0, dfb::in1);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        dfb_in0.wait_front(per_core_block_size);
        dfb_in1.wait_front(per_core_block_size);
        dfb_out.reserve_back(per_core_block_size);
        tile_regs_acquire();

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(dfb::in0, dfb::in1, i, i, i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, dfb::out);
        }
        tile_regs_release();

        dfb_in0.pop_front(per_core_block_size);
        dfb_in1.pop_front(per_core_block_size);
        dfb_out.push_back(per_core_block_size);
    }
}
