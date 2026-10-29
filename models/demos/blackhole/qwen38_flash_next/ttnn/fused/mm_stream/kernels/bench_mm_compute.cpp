// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 step-1 microbench: the cost of one tile matmul of the stock DRAM-sharded decode matmul's compute (fp32 DEST, the
// weight tile in srcA, the activation tile in srcB, matmul_block with a CT-tile-wide sub-block, K matmuls per sub-block
// accumulated in ONE DEST) as a function of fidelity (a compute-config property), weight format (the CB format) and CT.
// Defines: CT (1..4), KT (K tiles per sub-block), KBLK (the kt_dim passed to matmul_block, the stock's in0_block_w).
// Runtime arg 0 = reps of the sub-block.  No data movement: the CBs hold whatever the L1 holds.
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    const uint32_t reps = get_arg_val<uint32_t>(0);
    constexpr auto cb_in0 = tt::CBIndex::c_0;  // activation tiles, bf16
    constexpr auto cb_w = tt::CBIndex::c_1;    // weight tiles
    constexpr auto cb_out = tt::CBIndex::c_2;  // bf16 output tiles
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_w, cb_out);
    matmul_block_init(cb_in0, cb_w, false, CT, 1, KBLK);
    for (uint32_t rep = 0; rep < reps; ++rep) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < KT; ++k) {
            matmul_block(cb_in0, cb_w, k, (k % KBLK) * CT, 0, false, CT, 1, KBLK);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_block(0, cb_out, CT);
        tile_regs_release();
    }
}
