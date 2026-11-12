// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Int32 binary SFPU op (SFPU_OP_INIT_0 / SFPU_OP_CHAIN_0, e.g. gcd_tile / lcm_tile) on two Int32 tiles
// copied into DST[0] and DST[1], result packed from DST[2].
//
// An FPU binary op is initialised after the copies, as a fused FPU + SFPU kernel would, so the FPU
// binary init (which programs ADDR_MOD_3 with a Dest increment) and not a datacopy init is the last
// ADDR_MOD programmer before the SFPU op runs. The SFPU op must address Dest through its own slot.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/gcd.h"
#include "api/compute/lcm.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    const uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t tiles_per_iter = 1;

    constexpr uint32_t in0_id = tt::CBIndex::c_0;
    constexpr uint32_t in1_id = tt::CBIndex::c_1;
    constexpr uint32_t out_id = tt::CBIndex::c_16;
    CircularBuffer cb_in0(in0_id);
    CircularBuffer cb_in1(in1_id);
    CircularBuffer cb_out(out_id);

    compute_kernel_hw_startup(in0_id, in1_id, out_id);
    SFPU_OP_INIT_0

    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_in0.wait_front(tiles_per_iter);
        cb_in1.wait_front(tiles_per_iter);
        cb_out.reserve_back(tiles_per_iter);

        tile_regs_acquire();
        copy_init(in0_id);
        copy_tile(in0_id, /*in_tile_index=*/0, /*dst_tile_index=*/0);
        copy_init(in1_id);
        copy_tile(in1_id, /*in_tile_index=*/0, /*dst_tile_index=*/1);

        // FPU binary init after the copies: ADDR_MOD_3 now carries a Dest increment.
        add_init(in0_id, in1_id);

        SFPU_OP_CHAIN_0
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(/*dst_index=*/2, out_id);
        tile_regs_release();

        cb_out.push_back(tiles_per_iter);
        cb_in0.pop_front(tiles_per_iter);
        cb_in1.pop_front(tiles_per_iter);
    }
}
