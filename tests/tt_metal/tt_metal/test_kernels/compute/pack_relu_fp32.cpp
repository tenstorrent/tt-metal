// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// Copies NUM_TILES tiles through DEST with the packer ReLU set from RELU_CONFIG (ReluConfig::from_packed layout:
// bits [1:0] ReluType, bits [31:16] bf16 threshold).
void kernel_main() {
    constexpr uint32_t NUM_TILES = get_arg(args::NUM_TILES);
    constexpr uint32_t RELU_CONFIG = get_arg(args::RELU_CONFIG);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    pack_relu_config(ReluConfig::from_packed(RELU_CONFIG));

    for (uint32_t t = 0; t < NUM_TILES; ++t) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        copy_tile(dfb::in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
