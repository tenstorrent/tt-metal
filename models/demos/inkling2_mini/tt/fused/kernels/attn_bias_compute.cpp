// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, bias role, compute:
//   r    = rel * rel_scale                                   (fp32 SFPU)
//   s    = (r * r) @ G       G[c, c'] = 1 for c, c' < 16: per-head sum, broadcast over the head's 16 columns
//   rel  = r * rsqrt(max(s * inv, 1)) [* tau]  -> bf16
//   mask_j = bf16(rel @ P_j^T) + near_j (row broadcast)  -> bf16
#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
constexpr uint32_t cb_rel = 0, cb_rs = 1, cb_g = 2, cb_tau = 3, cb_p = 4, cb_near = 5;
constexpr uint32_t cb_r = 16, cb_sq = 17, cb_s = 18, cb_relo = 19, cb_b = 20, cb_mask = 21;

FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
FORCE_INLINE void put(uint32_t dst, uint32_t cb) {
    cb_reserve_back(cb, 1);
    pack_reconfig_data_format(cb);
    pack_tile(dst, cb);
    cb_push_back(cb, 1);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t inv = get_named_compile_time_arg_val("inv");  // fp32 bits
    constexpr uint32_t one = 0x3f800000u;

    compute_kernel_hw_startup(cb_rel, cb_r);
    cb_wait_front(cb_rel, 1);
    cb_wait_front(cb_rs, 1);
    tile_regs_acquire();
    load(cb_rel, 0, 0);
    load(cb_rs, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    mul_binary_tile(0, 0, 1);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_r);
    put(1, cb_sq);
    tile_regs_release();

    cb_wait_front(cb_sq, 1);
    cb_wait_front(cb_g, 1);
    cb_wait_front(cb_r, 1);
    if constexpr (has_tau) {
        cb_wait_front(cb_tau, 1);
    }
    tile_regs_acquire();
    reconfig_data_format(cb_g, cb_sq);  // matmul: srcA = in1
    matmul_init(cb_sq, cb_g);
    matmul_tiles(cb_sq, cb_g, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv);
    unary_max_tile_init();
    unary_max_tile(0, one);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if constexpr (has_tau) {
        load(cb_tau, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
    }
    load(cb_r, 0, 2);
    mul_binary_tile_init();
    mul_binary_tile(0, 2, 0);
    tile_regs_commit();
    tile_regs_wait();
    put(0, cb_relo);
    tile_regs_release();

    cb_wait_front(cb_relo, 1);
    cb_wait_front(cb_p, nj);
    cb_wait_front(cb_near, nj);
    for (uint32_t j = 0; j < nj; ++j) {
        tile_regs_acquire();
        reconfig_data_format(cb_p, cb_relo);
        matmul_init(cb_relo, cb_p, 1);
        matmul_tiles(cb_relo, cb_p, 0, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_b);
        tile_regs_release();

        cb_wait_front(cb_b, 1);
        tile_regs_acquire();
        reconfig_data_format(cb_b, cb_near);
        add_bcast_rows_init(cb_b, cb_near);
        add_tiles_bcast_rows(cb_b, cb_near, 0, j, 0);
        tile_regs_commit();
        tile_regs_wait();
        put(0, cb_mask);
        tile_regs_release();
        cb_pop_front(cb_b, 1);
    }
}
