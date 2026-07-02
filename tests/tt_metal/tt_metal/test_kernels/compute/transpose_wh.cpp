// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t NHtWt = get_arg(args::NHtWt);
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

#ifndef SHORT_INIT
    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);
#else
    unary_op_init_common(dfb::in, dfb::out);
    transpose_init(dfb::in);
#endif

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(dfb::in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
