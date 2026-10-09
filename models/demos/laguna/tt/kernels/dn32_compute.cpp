// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode down + expert sum (Laguna): DST j accumulates x_e @ Wd_e[:, c0 + j] over every active expert e
// (routing weights are already in x_e), then packs the CPC output tiles.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CPC = get_compile_time_arg_val(1);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t na = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (na == 0) {
        return;
    }
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t a = 0; a < na; ++a) {
        cb_wait_front(cb_x, Kt);
        cb_wait_front(cb_w, CPC * Kt);
        for (uint32_t j = 0; j < CPC; ++j) {
            for (uint32_t k = 0; k < Kt; ++k) {
                matmul_tiles(cb_x, cb_w, k, j * Kt + k, j);
            }
        }
        cb_pop_front(cb_x, Kt);
        cb_pop_front(cb_w, CPC * Kt);
    }
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, CPC);
    for (uint32_t j = 0; j < CPC; ++j) {
        pack_tile(j, cb_out);
    }
    cb_push_back(cb_out, CPC);
    tile_regs_release();
}
