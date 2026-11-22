// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention output prep, compute: out = bf16(attn * sigmoid(gate)), fp32 SFPU math (one dense page).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_a, cb_out);
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_g, 1);
    tile_regs_acquire();
    copy_init(cb_a);
    copy_tile(cb_a, 0, 0);
    copy_tile(cb_g, 0, 1);
    sigmoid_tile_init();
    sigmoid_tile(1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
