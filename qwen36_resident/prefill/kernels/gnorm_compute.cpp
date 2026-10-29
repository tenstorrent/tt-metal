// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// GDN output gate, compute: per group (32 tokens of one head, 4 tiles of 32 dims)
//   rs = rsqrt(mean over the 128 dims of o^2 + eps)      (row reduce of the 4 squared tiles)
//   y  = (o * rs) * silu(z)
// Compile-time args: 0 groups of this core, 1 eps (fp32 bits).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"

namespace {
constexpr uint32_t cb_o = 0, cb_z = 1, cb_scaler = 2, cb_sq = 3, cb_rs = 4, cb_on = 5, cb_sz = 6, cb_out = 7;

inline void pack_one(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    constexpr uint32_t groups = get_compile_time_arg_val(0);
    constexpr uint32_t eps = get_compile_time_arg_val(1);

    compute_kernel_hw_startup(cb_o, cb_o, cb_sq);
    cb_wait_front(cb_scaler, 1);
    for (uint32_t g = 0; g < groups; g++) {
        cb_wait_front(cb_o, 4);
        cb_wait_front(cb_z, 4);
        // o^2
        cb_reserve_back(cb_sq, 4);
        reconfig_data_format(cb_o, cb_o);
        pack_reconfig_data_format(cb_sq);
        mul_tiles_init(cb_o, cb_o);
        for (uint32_t d = 0; d < 4; d++) {
            tile_regs_acquire();
            mul_tiles(cb_o, cb_o, d, d, 0);
            pack_one(cb_sq);
        }
        cb_push_back(cb_sq, 4);
        cb_wait_front(cb_sq, 4);
        // rs
        cb_reserve_back(cb_rs, 1);
        reconfig_data_format(cb_scaler, cb_sq);
        pack_reconfig_data_format(cb_rs);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_rs);
        tile_regs_acquire();
        for (uint32_t d = 0; d < 4; d++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, d, 0, 0);
        }
        reduce_uninit();
        binop_with_scalar_tile_init();
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        pack_one(cb_rs);
        cb_push_back(cb_rs, 1);
        cb_pop_front(cb_sq, 4);
        cb_wait_front(cb_rs, 1);
        for (uint32_t d = 0; d < 4; d++) {
            // o * rs
            cb_reserve_back(cb_on, 1);
            reconfig_data_format(cb_o, cb_rs);
            pack_reconfig_data_format(cb_on);
            mul_bcast_cols_init(cb_o, cb_rs);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_o, cb_rs, d, 0, 0);
            pack_one(cb_on);
            cb_push_back(cb_on, 1);
            // silu(z)
            cb_reserve_back(cb_sz, 1);
            reconfig_data_format_srca(cb_z);
            pack_reconfig_data_format(cb_sz);
            copy_tile_to_dst_init_short(cb_z);
            tile_regs_acquire();
            copy_tile(cb_z, d, 0);
            silu_tile_init();
            silu_tile(0);
            pack_one(cb_sz);
            cb_push_back(cb_sz, 1);
            // product
            cb_wait_front(cb_on, 1);
            cb_wait_front(cb_sz, 1);
            cb_reserve_back(cb_out, 1);
            reconfig_data_format(cb_on, cb_sz);
            pack_reconfig_data_format(cb_out);
            mul_tiles_init(cb_on, cb_sz);
            tile_regs_acquire();
            mul_tiles(cb_on, cb_sz, 0, 0, 0);
            pack_one(cb_out);
            cb_push_back(cb_out, 1);
            cb_pop_front(cb_on, 1);
            cb_pop_front(cb_sz, 1);
        }
        cb_pop_front(cb_rs, 1);
        cb_pop_front(cb_o, 4);
        cb_pop_front(cb_z, 4);
    }
}
