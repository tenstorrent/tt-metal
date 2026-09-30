// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_unary/softplus.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t tiles = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(0, 1);
    copy_init(0);
    softplus_tile_init();
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        cb_wait_front(0, 1);
        cb_reserve_back(1, 1);
        tile_regs_acquire();
        copy_tile(0, 0, 0);
        softplus_tile(0, 0x3f800000, 0x3f800000, 0x41a00000);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 1);
        tile_regs_release();
        cb_pop_front(0, 1);
        cb_push_back(1, 1);
    }
}
