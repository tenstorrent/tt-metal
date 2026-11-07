// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dot-product finish, compute (one core, SFPU, fp32 in DEST): adds the pages of per-core values lane by lane (unused
// lanes are zero), then reduces the 1024 lanes to one value (columns, then rows) at element (0, 0). do_sqrt (runtime
// argument): then takes its square root.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/sqrt.h"

void kernel_main() {
    const uint32_t n = get_arg_val<uint32_t>(0), do_sqrt = get_arg_val<uint32_t>(1);
    compute_kernel_hw_startup(CB_A, CB_OUT);
    tile_regs_acquire();
    copy_init(CB_A);
    add_binary_tile_init();
    for (uint32_t t = 0; t < n; ++t) {
        cb_wait_front(CB_A, 1);
        copy_tile(CB_A, 0, t == 0 ? 0 : 1);
        cb_pop_front(CB_A, 1);
        if (t) {
            add_binary_tile(0, 1, 0);
        }
    }
    sfpu_reduce_init<PoolType::SUM, DataFormat::Float32>();
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_COL>(0);
    sfpu_reduce<PoolType::SUM, DataFormat::Float32, ReduceDim::REDUCE_ROW>(0);
    if (do_sqrt) {
        sqrt_tile_init();
        sqrt_tile(0);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(CB_OUT, 1);
    pack_tile(0, CB_OUT);
    cb_push_back(CB_OUT, 1);
    tile_regs_release();
}
