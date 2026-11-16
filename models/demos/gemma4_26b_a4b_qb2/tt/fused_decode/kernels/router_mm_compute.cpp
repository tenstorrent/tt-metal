// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Batch-1 router projection, matmul core compute (fused_decode/router.py: RouterB1): the `kt` FP32 x BF16 tile products
// accumulated in FP32 DST (HiFi4), packed FP32 (row 0 = the K-slice partial of 32 logits).

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);  // matmul: in0 -> SrcB, in1 -> SrcA
    matmul_init(cb_x, cb_w);
    cb_wait_front(cb_x, kt);
    cb_wait_front(cb_w, kt);
    tile_regs_acquire();
    for (uint32_t t = 0; t < kt; ++t) {
        matmul_tiles(cb_x, cb_w, t, t, 0);
    }
    tile_regs_commit();
    cb_reserve_back(cb_out, 1);
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();
    cb_push_back(cb_out, 1);
}
