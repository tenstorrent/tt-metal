// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/dropout.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

// dropout_sfpu.cpp over the 2.0 host API (DataflowBuffer bindings, named compile-time args).
void kernel_main() {
    constexpr uint32_t num_tiles = get_arg(args::num_tiles);
    constexpr uint32_t seed = get_arg(args::seed);
    constexpr uint32_t int_probability = get_arg(args::int_probability);
    constexpr uint32_t int_scale_factor = get_arg(args::int_scale_factor);
    // Produce the tile dropout works on with an FPU add (in + in = 2x) instead of a datacopy, so that an FPU
    // binary init is the last ADDR_MOD_3 programmer before dropout_tile.
    constexpr bool fpu_add_input = get_arg(args::fpu_add_input) != 0;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);
    const uint32_t in_id = dfb_in.get_id();
    const uint32_t out_id = dfb_out.get_id();

    compute_kernel_hw_startup(in_id, in_id, out_id);
    if constexpr (fpu_add_input) {
        add_init(in_id, in_id);
    } else {
        copy_init(in_id);
    }
    dropout_kernel_init(seed);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        dfb_out.reserve_back(1);
        tile_regs_acquire();
        dfb_in.wait_front(1);
        if constexpr (fpu_add_input) {
            add_tiles(in_id, in_id, 0, 0, 0);
        } else {
            copy_tile(in_id, 0, 0);
        }
        dropout_tile(0, int_probability, int_scale_factor);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, out_id);
        dfb_in.pop_front(1);
        tile_regs_release();
        dfb_out.push_back(1);
    }
}
