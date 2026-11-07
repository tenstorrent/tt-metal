// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// GatedDeltaNet prefill output, compute: per item, out = o / sqrt(mean_row(o^2) + eps) * w * silu(z) over the head's
// 128 values (4 tiles per row): row sums of o^2 by a matmul with all-ones, w as row tiles (row-broadcast).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "../../ternary/kernels/core_span.hpp"

namespace {
constexpr uint32_t cb_o = 0, cb_z = 1, cb_c = 2, cb_sq = 3, cb_inv = 4, cb_zw = 5, cb_out = 16;
constexpr uint32_t ONES = 4;
constexpr auto TO_A = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void put(uint32_t cb, uint32_t n) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    cb_reserve_back(cb, n);
    for (uint32_t d = 0; d < n; ++d) {
        pack_tile<true>(d, cb, d);
    }
    cb_push_back(cb, n);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    // common args: Nitems, ncores, grid_x, 1/128 (fp32 bits), eps (fp32 bits)
    uint32_t first, n;
    core_span(
        get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2), 1, first, n);
    const uint32_t inv_d = get_common_arg_val<uint32_t>(3), eps = get_common_arg_val<uint32_t>(4);
    compute_kernel_hw_startup(cb_o, cb_o, cb_sq);
    cb_wait_front(cb_c, 5);
    for (uint32_t it = 0; it < n; ++it) {
        cb_wait_front(cb_o, 4);
        cb_wait_front(cb_z, 4);
        // squares
        tile_regs_acquire();
        fmt(cb_o, cb_o);
        mul_init(cb_o, cb_o);
        for (uint32_t d = 0; d < 4; ++d) {
            mul_tiles(cb_o, cb_o, d, d, d);
        }
        put(cb_sq, 4);
        cb_wait_front(cb_sq, 4);
        // inv = rsqrt(sum / 128 + eps), every column
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(cb_sq, cb_c);
        matmul_init(cb_sq, cb_c);
        for (uint32_t d = 0; d < 4; ++d) {
            matmul_tiles(cb_sq, cb_c, d, ONES, 0);
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_d);
        add_unary_tile(0, eps);
        rsqrt_tile_init();
        rsqrt_tile(0);
        put(cb_inv, 1);
        cb_pop_front(cb_sq, 4);
        cb_wait_front(cb_inv, 1);
        // zw = silu(z) * w (w row tiles broadcast down the rows)
        tile_regs_acquire();
        fmt(cb_z, cb_z);
        copy_init(cb_z);
        for (uint32_t d = 0; d < 4; ++d) {
            copy_tile(cb_z, d, d);
        }
        silu_tile_init();
        for (uint32_t d = 0; d < 4; ++d) {
            silu_tile(d);
        }
        put(cb_sq, 4);  // silu(z), fp32
        cb_wait_front(cb_sq, 4);
        tile_regs_acquire();
        fmt(cb_sq, cb_c);
        mul_bcast_rows_init(cb_sq, cb_c);
        for (uint32_t d = 0; d < 4; ++d) {
            mul_tiles_bcast_rows(cb_sq, cb_c, d, d, d);
        }
        put(cb_zw, 4);
        cb_pop_front(cb_sq, 4);
        cb_wait_front(cb_zw, 4);
        // out = (o * inv) * zw
        tile_regs_acquire();
        fmt(cb_o, cb_inv);
        mul_bcast_cols_init(cb_o, cb_inv);
        for (uint32_t d = 0; d < 4; ++d) {
            mul_tiles_bcast_cols(cb_o, cb_inv, d, 0, d);
        }
        fmt(cb_zw, cb_zw);
        mul_reuse_dest_init<TO_A>(cb_zw);
        for (uint32_t d = 0; d < 4; ++d) {
            mul_reuse_dest_tiles<TO_A>(cb_zw, d, d);
        }
        put(cb_out, 4);
        cb_pop_front(cb_zw, 4);
        cb_pop_front(cb_inv, 1);
        cb_pop_front(cb_o, 4);
        cb_pop_front(cb_z, 4);
    }
}
