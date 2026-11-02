// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/matmul.h"
#include "api/dataflow/circular_buffer.h"

// One matmul_init, then two products of the same tiles: in0 @ in1 -> [exp] -> pack, and in0 @ in1 -> pack, with no
// second matmul_init. With EXP_BETWEEN the second product runs after an exp_tile_init / exp_tile, so it must not
// depend on anything the exp leaves behind (the FPU's replay slots, which hold matmul's recorded MVMUL sequence).
void kernel_main() {
    constexpr uint32_t in0 = tt::CBIndex::c_0;
    constexpr uint32_t in1 = tt::CBIndex::c_1;
    constexpr uint32_t out = tt::CBIndex::c_16;
    CircularBuffer cb_in0(in0);
    CircularBuffer cb_in1(in1);
    CircularBuffer cb_out(out);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out);
    matmul_init(in0, in1);
    cb_in0.wait_front(1);
    cb_in1.wait_front(1);
    cb_out.reserve_back(2);

    tile_regs_acquire();
    matmul_tiles(in0, in1, 0, 0, 0);
#ifdef EXP_BETWEEN
    exp_tile_init();
    exp_tile(0);
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();

    tile_regs_acquire();
    matmul_tiles(in0, in1, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, out);
    tile_regs_release();

    cb_out.push_back(2);
    cb_in0.pop_front(1);
    cb_in1.pop_front(1);
}
