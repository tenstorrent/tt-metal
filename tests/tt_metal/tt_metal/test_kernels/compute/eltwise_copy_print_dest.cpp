// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
namespace NAMESPACE {

void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    unary_op_init_common(tt::CB::c_in0);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst(tt::DstMode::Half);

        // Pop tile after tile, copy to DST and pack
        cb_wait_front(tt::CB::c_in0, 1);
        cb_reserve_back(tt::CB::c_out0, 1);
        copy_tile(tt::CB::c_in0, 0, 0);

        dprint_tensix_dest_reg();

        pack_tile(0, tt::CB::c_out0);

        cb_pop_front(tt::CB::c_in0, 1);
        cb_push_back(tt::CB::c_out0, 1);

        release_dst(tt::DstMode::Half);
    }
}
}  // namespace NAMESPACE
