// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute for attn_nr, one head per core, fp32 DST:
//   n_t   = x_t * w_t * rsqrt(mean(x^2) + eps)                      (RMSNorm, weight already +1 pre-offset)
//   out_t = n_t * cos_t + n_p * sin_t * (t < Rt/2 ? -1 : +1)        for the Rt rotary tiles (p = partner tile,
//           i.e. rotate_half pairs dims i and i + 32*Rt/2), out_t = n_t for the rest.
// sum(x^2) is x . x^T (transposed-B matmul: [0,0] is the row-0 dot product); the per-row scale comes from column 0
// of the rinv tile (sfpu_mul_bcast_col). CBs alternate bf16 (x, w, cos/sin, out) and fp32 (ss, rinv, n), so srcA's
// format is reconfigured before every copy and the packer's before each output CB change.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_x = 0, cb_w = 1, cb_cs = 2, cb_ss = 3, cb_r = 4, cb_n = 5, cb_out = 6;

namespace {
void copy_to(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr uint32_t Rt = get_compile_time_arg_val(1);
    constexpr uint32_t half = Rt / 2;
    const uint32_t inv_d_bits = get_arg_val<uint32_t>(0);
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_x, cb_x, cb_ss);
    CircularBuffer cx(cb_x), cw(cb_w), ccs(cb_cs), css(cb_ss), cr(cb_r), cn(cb_n), co(cb_out);
    cx.wait_front(Dt);
    cw.wait_front(Dt);
    ccs.wait_front(2 * Rt);

    // ss[0,0] = sum over the Dt tiles of row 0 squared
    css.reserve_back(1);
    reconfig_data_format(cb_x, cb_x);
    matmul_init(cb_x, cb_x, 1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < Dt; t++) {
        matmul_tiles(cb_x, cb_x, t, t, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_ss);
    pack_tile(0, cb_ss);
    tile_regs_release();
    css.push_back(1);
    css.wait_front(1);

    // rinv, once
    cr.reserve_back(1);
    tile_regs_acquire();
    copy_to(cb_ss, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, inv_d_bits);
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_r);
    tile_regs_release();
    cr.push_back(1);
    cr.wait_front(1);

    // n_t = x_t * w_t * rinv
    cn.reserve_back(Dt);
    for (uint32_t t = 0; t < Dt; t++) {
        tile_regs_acquire();
        copy_to(cb_x, t, 0);
        copy_to(cb_w, t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        copy_to(cb_r, 0, 2);
        sfpu_mul_bcast_col_init();
        sfpu_mul_bcast_col(0, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_n);
        tile_regs_release();
    }
    cn.push_back(Dt);
    cn.wait_front(Dt);

    // RoPE on the rotary tiles, pass-through for the rest; bf16 out
    co.reserve_back(Dt);
    pack_reconfig_data_format(cb_out);
    for (uint32_t t = 0; t < Dt; t++) {
        tile_regs_acquire();
        if (t < Rt) {
            const uint32_t p = (t < half) ? t + half : t - half;
            copy_to(cb_n, t, 0);
            copy_to(cb_cs, t, 1);  // cos_t
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);
            copy_to(cb_n, p, 1);
            copy_to(cb_cs, Rt + t, 2);  // sin_t
            mul_binary_tile_init();
            mul_binary_tile(1, 2, 1);
            if (t < half) {
                sub_binary_tile_init();
                sub_binary_tile(0, 1, 0);
            } else {
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        } else {
            copy_to(cb_n, t, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    co.push_back(Dt);
    cx.pop_front(Dt);
    cw.pop_front(Dt);
    ccs.pop_front(2 * Rt);
    css.pop_front(1);
    cr.pop_front(1);
    cn.pop_front(Dt);
}
