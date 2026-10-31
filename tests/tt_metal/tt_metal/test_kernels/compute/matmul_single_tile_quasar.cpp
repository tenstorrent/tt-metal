// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"

// One-tile matmul, in0 x in1 -> out. Float operands run at the kernel's MATH_FIDELITY; with Int8 operands the
// Quasar LLK runs one phase whatever MATH_FIDELITY asks for, which the HiFi4 case checks.
void kernel_main() {
    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::out);
    matmul_init(dfb::in0, dfb::in1);

    dfb_out.reserve_back(1);
    dfb_in0.wait_front(1);
    dfb_in1.wait_front(1);
    tile_regs_acquire();
    matmul_tiles(dfb::in0, dfb::in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::out);
    tile_regs_release();
    dfb_in0.pop_front(1);
    dfb_in1.pop_front(1);
    dfb_out.push_back(1);
}
