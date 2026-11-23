// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Prefill attention sink, compute: per row, out_h = q tile x S_h (HiFi4; one bf16 term per element, exact).
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"

void kernel_main() {
    constexpr uint32_t H = get_named_compile_time_arg_val("h");
    constexpr uint32_t cb_q = 0, cb_sel = 1, cb_out = 16;
    const uint32_t n = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(cb_q, cb_sel, cb_out);
    matmul_init(cb_q, cb_sel);
    cb_wait_front(cb_sel, H);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(cb_q, 1);
        for (uint32_t h = 0; h < H; ++h) {
            tile_regs_acquire();
            matmul_tiles(cb_q, cb_sel, 0, h, 0);
            tile_regs_commit();
            tile_regs_wait();
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_release();
        }
        cb_pop_front(cb_q, 1);
    }
    cb_pop_front(cb_sel, H);
}
