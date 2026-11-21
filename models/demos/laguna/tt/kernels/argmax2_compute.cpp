// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Greedy pick, stage 1 (tile-engine version), compute: per vocabulary tile, the max over each row (reduce MAX along
// the row; column 0 of the packed tile holds row r's max, exact for bf16) -> cb_max, one tile per input tile.

#include <cstdint>
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reduce.h"
#include "api/compute/common.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t S = get_compile_time_arg_val(2);
    constexpr uint32_t per = get_compile_time_arg_val(3);
    constexpr uint32_t W = get_compile_time_arg_val(4);
    constexpr uint32_t cb_in = 0, cb_scaler = 1, cb_max = 16;
    // this unit's tile count (the reader's rule; the last unit of a tile row may have fewer, idle cores none)
    const uint32_t w = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t j0 = (w % S) * per;
    const uint32_t n = w >= W || j0 >= Vt ? 0 : (j0 + per < Vt ? per : Vt - j0);
    if (n == 0) {
        return;
    }
    compute_kernel_hw_startup(cb_in, cb_scaler, cb_max);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_max);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t j = 0; j < n; ++j) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, 0, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_max, 1);
        pack_tile(0, cb_max);
        cb_push_back(cb_max, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
    reduce_uninit();
    cb_pop_front(cb_scaler, 1);
}
