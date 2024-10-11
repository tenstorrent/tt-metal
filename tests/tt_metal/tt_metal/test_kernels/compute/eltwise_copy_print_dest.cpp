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
    bool remap = get_compile_time_arg_val(1) != 0;
    bool swizzle = get_compile_time_arg_val(2) != 0;

    unary_op_init_common(tt::CB::c_in0);
#ifdef ARCH_BLACKHOLE
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_remap_addrs_RMW>(remap);
    cfg_reg_rmw_tensix<DEST_ACCESS_CFG_swizzle_32b_RMW>(swizzle);
#endif
    acquire_dst();
    cb_wait_front(tt::CB::c_in0, per_core_tile_cnt);
    cb_reserve_back(tt::CB::c_out0, per_core_tile_cnt);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        copy_tile(tt::CB::c_in0, b, b);
        dprint_tensix_dest_reg(b);
    }

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        pack_tile(b, tt::CB::c_out0);
        cb_pop_front(tt::CB::c_in0, 1);
        cb_push_back(tt::CB::c_out0, 1);
    }

    release_dst();
}
}  // namespace NAMESPACE
