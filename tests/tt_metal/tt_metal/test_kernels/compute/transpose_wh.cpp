// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    uint32_t NHtWt = get_compile_time_arg_val(0);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

#ifndef SHORT_INIT
    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    transpose_wh_init_short(tt::CBIndex::c_0);
#endif

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tile_regs_acquire();
        transpose_wh_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        cb16.push_back(1);
        cb0.pop_front(1);
    }
}
}  // namespace NAMESPACE
