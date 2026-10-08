// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 compute for test_eltwise_add_quasar: C = A + B, one tile at a time, on one Tensix engine.
// The Metal 2.0 counterpart of programming_examples/eltwise_binary/kernels/compute/tiles_add.cpp.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t dst_reg = 0;

    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    add_init(dfb::in0, dfb::in1);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in0.wait_front(1);
        dfb_in1.wait_front(1);

        tile_regs_acquire();
        add_tiles(dfb::in0, dfb::in1, 0, 0, dst_reg);
        tile_regs_commit();

        dfb_out.reserve_back(1);
        tile_regs_wait();
        pack_tile(dst_reg, dfb::out);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in0.pop_front(1);
        dfb_in1.pop_front(1);
    }
}
