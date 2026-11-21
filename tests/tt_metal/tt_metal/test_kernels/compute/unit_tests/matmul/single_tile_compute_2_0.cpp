// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

// Single-tile matmul, out = in0 * in1, over the 2.0 host API (DataflowBuffer bindings).
void kernel_main() {
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in0_id = dfb_in0.get_id();
    const uint32_t in1_id = dfb_in1.get_id();
    const uint32_t out_id = dfb_out.get_id();

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_id, in1_id, out_id);
    matmul_init(in0_id, in1_id);

    dfb_out.reserve_back(1);
    tile_regs_acquire();
    dfb_in0.wait_front(1);
    dfb_in1.wait_front(1);
    matmul_tiles(in0_id, in1_id, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out_id);
    dfb_in0.pop_front(1);
    dfb_in1.pop_front(1);
    tile_regs_release();
    dfb_out.push_back(1);
}
