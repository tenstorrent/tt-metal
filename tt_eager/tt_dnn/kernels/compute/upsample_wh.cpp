// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {

    uint32_t NHtWt = get_compile_time_arg_val(0);
    unary_op_init_common(tt::CB::c_in0);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_wait_front(tt::CB::c_in0, 1);
        cb_reserve_back(tt::CB::c_out0, 1);

        acquire_dst(tt::DstMode::Half);
        copy_tile_with_upscale(tt::CB::c_in0, 0, 0);
        pack_tile(0, tt::CB::c_out0);
        release_dst(tt::DstMode::Half);

        cb_push_back(tt::CB::c_out0, 1);
        cb_pop_front(tt::CB::c_in0, 1);
    }
}
}
