// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode down + expert sum (Laguna): DST j accumulates x_e @ Wd_e[:, c0 + j] over every active expert e
// (routing weights are already in x_e), then packs the CPC output tiles.

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t CPC = get_compile_time_arg_val(1);
    constexpr uint32_t Kt_sh = get_compile_time_arg_val(2);  // shared expert unit (on one group's cores), cb_w_sh
    constexpr uint32_t KS = get_compile_time_arg_val(3);     // units per routed expert (Kt / KS K tiles each)
    constexpr uint32_t Kq = Kt / KS;
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_w_sh = 6, cb_out = 16;
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x, cb_w, cb_out);
    cb_wait_front(cb_meta, 1);
    const uint32_t na = read_tile_value(cb_meta, 0, 0);
    const uint32_t sh = Kt_sh > 0 ? read_tile_value(cb_meta, 0, 1) : 0;
    cb_pop_front(cb_meta, 1);
    if (na == 0 && sh == 0) {
        return;
    }
    matmul_init(cb_x, cb_w);
    tile_regs_acquire();
    for (uint32_t a = 0; a < na; ++a) {
        cb_wait_front(cb_x, Kq);
        cb_wait_front(cb_w, CPC * Kq);
        for (uint32_t j = 0; j < CPC; ++j) {
            for (uint32_t k = 0; k < Kq; ++k) {
                matmul_tiles(cb_x, cb_w, k, j * Kq + k, j);
            }
        }
        cb_pop_front(cb_x, Kq);
        cb_pop_front(cb_w, CPC * Kq);
    }
    if constexpr (Kt_sh > 0) {
        if (sh) {
            reconfig_data_format(cb_w_sh, cb_x);
            matmul_init(cb_x, cb_w_sh);
            cb_wait_front(cb_x, Kt_sh);
            cb_wait_front(cb_w_sh, CPC * Kt_sh);
            for (uint32_t j = 0; j < CPC; ++j) {
                for (uint32_t k = 0; k < Kt_sh; ++k) {
                    matmul_tiles(cb_x, cb_w_sh, k, j * Kt_sh + k, j);
                }
            }
            cb_pop_front(cb_x, Kt_sh);
            cb_pop_front(cb_w_sh, CPC * Kt_sh);
        }
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
