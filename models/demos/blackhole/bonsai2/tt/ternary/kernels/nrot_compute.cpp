// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Decode RMSNorm + rotation, compute. Sum of squares over the raw pages (masked to the face row 0 values), every
// element = ONES @ (sum (P^2 * OM) @ ONES); inv = rsqrt(sum / K + eps). Own block: xn = X_b * inv * W_b (untilized for
// the writer: the normalised activation) and xd = X_b * inv * WD_b (signs folded into the weight), rotated as
// rot_compute.cpp: Y = H32 xd H32/32.
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "core_span.hpp"

namespace {
constexpr uint32_t cb_x = 0, cb_c = 2, cb_sc = 3, cb_st = 23, cb_xt = 24, cb_sq = 25, cb_r = 26, cb_inv = 27,
                   cb_xd = 28;
constexpr uint32_t cb_t = 29;
constexpr uint32_t cb_xn = 17, cb_y = 16;
constexpr uint32_t H = 0, H32 = 1, OM = 2, WB = 3, WDB = 4, ONES = 5;  // OM: the raw-order row-0 mask
constexpr auto TO_A = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
ALWI void fmt(uint32_t a, uint32_t b) { reconfig_data_format(a, b); }
ALWI void fmt_mm(uint32_t a, uint32_t b) { reconfig_data_format<SrcOrder::Reverse>(a, b); }
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
ALWI void put_untilized(uint32_t cb) {
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_untilize_dest_init<1>(cb);
    cb_reserve_back(cb, 1);
    pack_untilize_dest<1>(cb);
    cb_push_back(cb, 1);
    tile_regs_release();
    pack_untilize_uninit(cb);
}
}  // namespace

void kernel_main() {
    // common args: grid_x, B, 1/K (fp32 bits), eps (fp32 bits)
    const uint32_t b = core_index(get_common_arg_val<uint32_t>(0));
    const uint32_t B = get_common_arg_val<uint32_t>(1);
    const uint32_t inv_k = get_common_arg_val<uint32_t>(2), eps = get_common_arg_val<uint32_t>(3);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_c, cb_xt, cb_xt);

    // own block, 32 x 32
    tilize_init(cb_x, 1, cb_xt);
    cb_wait_front(cb_x, 1);
    cb_reserve_back(cb_xt, 1);
    tilize_block(cb_x, 1, cb_xt);
    cb_push_back(cb_xt, 1);
    cb_pop_front(cb_x, 1);
    tilize_uninit(cb_x, cb_xt);
    // Sum of squares straight from the raw pages (no tilize: a sum does not care about element order): each page
    // is unpacked as if it were a tile, squared, and masked with RM (1 where the raw element is a face row 0 value,
    // i.e. raw index % 32 < 16, laid out in tile order), then summed with ONES.
    cb_wait_front(cb_xt, 1);
    cb_wait_front(cb_c, 6);
    for (uint32_t i = 0; i < 2 * B; i += 8) {
        const uint32_t n = 2 * B - i < 8 ? 2 * B - i : 8;
        cb_wait_front(cb_sc, i + n);
        tile_regs_acquire();
        fmt(cb_sc, cb_sc);
        mul_init(cb_sc, cb_sc);
        for (uint32_t k = 0; k < n; ++k) {
            mul_tiles(cb_sc, cb_sc, i + k, i + k, k);
        }
        fmt(cb_c, cb_c);
        mul_reuse_dest_init<TO_A>(cb_c);
        for (uint32_t k = 0; k < n; ++k) {
            mul_reuse_dest_tiles<TO_A>(cb_c, OM, k);
        }
        put(cb_sq, n);
    }
    cb_pop_front(cb_sc, 2 * B);
    cb_wait_front(cb_sq, 2 * B);
    tile_regs_acquire();
    fmt_mm(cb_sq, cb_c);
    matmul_init(cb_sq, cb_c);
    for (uint32_t k = 0; k < 2 * B; ++k) {
        matmul_tiles(cb_sq, cb_c, k, ONES, 0);
    }
    put(cb_r, 1);
    cb_pop_front(cb_sq, 2 * B);
    cb_wait_front(cb_r, 1);
    // total in every element: ONES @ R (R[r][c] = row r's sum, the same for every c)
    tile_regs_acquire();
    fmt_mm(cb_c, cb_r);
    matmul_init(cb_c, cb_r);
    matmul_tiles(cb_c, cb_r, ONES, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_k);
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    put(cb_inv, 1);
    cb_pop_front(cb_r, 1);
    cb_wait_front(cb_inv, 1);
    // xn (normalised, untilized for the writer) and xd (with the signs, rotated below)
    tile_regs_acquire();
    fmt(cb_xt, cb_inv);
    mul_init(cb_xt, cb_inv);
    mul_tiles(cb_xt, cb_inv, 0, 0, 0);
    fmt(cb_c, cb_c);
    mul_reuse_dest_init<TO_A>(cb_c);
    mul_reuse_dest_tiles<TO_A>(cb_c, WB, 0);
    put_untilized(cb_xn);
    tile_regs_acquire();
    fmt(cb_xt, cb_inv);
    mul_init(cb_xt, cb_inv);
    mul_tiles(cb_xt, cb_inv, 0, 0, 0);
    fmt(cb_c, cb_c);
    mul_reuse_dest_init<TO_A>(cb_c);
    mul_reuse_dest_tiles<TO_A>(cb_c, WDB, 0);
    put(cb_xd, 1);
    cb_pop_front(cb_inv, 1);
    cb_pop_front(cb_xt, 1);
    cb_wait_front(cb_xd, 1);
    // T = H32 @ xd, Y = T @ (H32 / 32)
    tile_regs_acquire();
    fmt_mm(cb_c, cb_xd);
    matmul_init(cb_c, cb_xd);
    matmul_tiles(cb_c, cb_xd, H, 0, 0);
    put(cb_t, 1);
    cb_wait_front(cb_t, 1);
    tile_regs_acquire();
    fmt_mm(cb_t, cb_c);
    matmul_init(cb_t, cb_c);
    matmul_tiles(cb_t, cb_c, 0, H32, 0);
    put_untilized(cb_y);
    cb_pop_front(cb_t, 1);
    cb_pop_front(cb_xd, 1);
    cb_pop_front(cb_c, 6);
}
