// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/typecast.h"
void kernel_main() {
    compute_kernel_hw_startup(0, 1, 3);
    const uint32_t count = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < count; ++i) {
        cb_wait_front(0, 1);
        cb_wait_front(1, 1);
        cb_wait_front(2, 1);
        tile_regs_acquire();
        copy_init(0);
        copy_tile(0, 0, 0);
        copy_init(1);
        copy_tile(1, 0, 1);
        copy_init(2);
        copy_tile(2, 0, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        cb_reserve_back(3, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, 3);
        tile_regs_release();
        cb_push_back(3, 1);
        cb_pop_front(0, 1);
        cb_pop_front(1, 1);
        cb_pop_front(2, 1);
    }
}
