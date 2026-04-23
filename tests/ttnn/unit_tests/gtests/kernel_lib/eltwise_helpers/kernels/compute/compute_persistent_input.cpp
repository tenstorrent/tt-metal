// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Persistent B input: FpuAdd with WaitUpfrontNoPop for B.
// B tiles (cb_in1) are waited upfront before the first block and never popped — they
// persist in the CB and are reused by both blocks. A tiles stream per tile normally.
// Both blocks write sequentially to cb_out, total 2*tiles_per_block output tiles.
//
// Validates: B tiles survive across two eltwise_op calls with WaitUpfrontNoPop.
// Runtime args: [0] tiles_per_block

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

void kernel_main() {
    // Host passes total A tiles (2 blocks); divide to get per-block count
    const uint32_t total_a_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tiles_per_block = total_a_tiles / 2;
    if (tiles_per_block == 0) {
        return;
    }

    constexpr auto cb_in0 = tt::CBIndex::c_0;   // A tiles (streamed per tile, 2 blocks)
    constexpr auto cb_in1 = tt::CBIndex::c_1;   // B tiles (persistent, WaitUpfrontNoPop)
    constexpr auto cb_out = tt::CBIndex::c_16;  // all output tiles

    using namespace compute_kernel_lib;

    auto chain = sfpu_chain(FpuAdd<
                            cb_in0,
                            cb_in1,
                            Dst::D0,
                            BroadcastDim::NONE,
                            BinaryInputPolicy::WaitAndPopPerTile,
                            BinaryInputPolicy::WaitUpfrontNoPop>{});

    // Block 0: A[0..tiles_per_block-1] + B → cb_out tiles 0..tiles_per_block-1
    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
    // Block 1: A[tiles_per_block..2*tiles_per_block-1] + B → cb_out tiles tiles_per_block..2*tiles_per_block-1
    // B tiles still in cb_in1 (WaitUpfrontNoPop never popped them)
    eltwise_op<cb_out>(chain, EltwiseTileShape::flat(tiles_per_block));
}
