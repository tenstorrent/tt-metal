// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Message-gate combine, compute, per column tile j of this core (row-form tiles, rows = gated vectors):
//   zb = bf16(z) [* block mask]           (HF: bf16 down_proj output)
//   f = tanh(zb @ F[:, j]), g = tanh(zb @ G[:, j])     (fp32 projections, FPU reads F / G as tf32)
//   out = h - f * h + g * m               (SFPU, fp32)
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/debug/kernel_profiler.hpp"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t kz = get_named_compile_time_arg_val("kz");
    constexpr uint32_t nt = get_named_compile_time_arg_val("nt");
    constexpr uint32_t has_mask = get_named_compile_time_arg_val("has_mask");
    constexpr uint32_t cb_z = 0, cb_mask = 1, cb_f = 2, cb_g = 3, cb_h = 4, cb_m = 5, cb_zb = 16, cb_out = 17;

    compute_kernel_hw_startup(cb_z, cb_zb);
    cb_wait_front(cb_z, kz);
    if constexpr (has_mask) {
        cb_wait_front(cb_mask, kz);
    }
    pack_reconfig_data_format(cb_zb);
    cb_reserve_back(cb_zb, kz);
    for (uint32_t kt = 0; kt < kz; ++kt) {
        tile_regs_acquire();
        load(cb_z, kt, 0);
        if constexpr (has_mask) {
            load(cb_mask, kt, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_zb);
        tile_regs_release();
    }
    cb_push_back(cb_zb, kz);

    cb_wait_front(cb_zb, kz);
    {
        DeviceZoneScopedN("C_wait_fg");
        cb_wait_front(cb_f, kz * nt);
        cb_wait_front(cb_g, kz * nt);
        cb_wait_front(cb_h, nt);
        cb_wait_front(cb_m, nt);
    }
    DeviceZoneScopedN("C_math");
    pack_reconfig_data_format(cb_out);
    cb_reserve_back(cb_out, nt);
    for (uint32_t j = 0; j < nt; ++j) {
        tile_regs_acquire();
        // matmul: in0 -> SrcB, in1 -> SrcA
        reconfig_data_format(cb_f, cb_zb);
        matmul_init(cb_zb, cb_f);
        for (uint32_t kt = 0; kt < kz; ++kt) {
            matmul_tiles(cb_zb, cb_f, kt, kt * nt + j, 1);
        }
        reconfig_data_format(cb_g, cb_zb);
        matmul_init(cb_zb, cb_g);
        for (uint32_t kt = 0; kt < kz; ++kt) {
            matmul_tiles(cb_zb, cb_g, kt, kt * nt + j, 2);
        }
        tanh_tile_init();
        tanh_tile(1);
        tanh_tile(2);
        load(cb_h, j, 0);
        load(cb_m, j, 3);
        mul_binary_tile_init();
        mul_binary_tile(1, 0, 1);  // f * h
        mul_binary_tile(2, 3, 2);  // g * m
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);  // h - f * h
        add_binary_tile_init();
        add_binary_tile(0, 2, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, nt);
}
