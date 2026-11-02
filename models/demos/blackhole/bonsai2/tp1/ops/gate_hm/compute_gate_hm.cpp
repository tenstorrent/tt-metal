// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Compute for gate_hm: gated = o * rsqrt(mean(o^2) + eps) * w * silu(z), one head per core, in fp32 DST (the chain
// it replaces: rms_norm(o, weight=w, eps) on [heads, 1, D], then silu(z) and a multiply, three ops).
// sum(o^2) is o . o^T (a matmul against the transposed tile: [0,0] is the row-0 dot product); the per-row scale
// is applied by sfpu_mul_bcast_col from column 0 of the rinv tile (row 0 carries it).

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

constexpr uint32_t cb_o = 0, cb_z = 1, cb_w = 2, cb_ss = 3, cb_out = 4;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    // Debug stage (test only): 0 = o, 1 = o*w, 2 = o*w*rinv, 3 = the full gate (default).
    constexpr uint32_t stage = get_compile_time_arg_val(1);
    const uint32_t inv_d_bits = get_arg_val<uint32_t>(0);  // 1/D as fp32 bits
    const uint32_t eps_bits = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup(cb_o, cb_o, cb_ss);
    CircularBuffer co(cb_o), cz(cb_z), cw(cb_w), css(cb_ss), cout(cb_out);
    co.wait_front(Dt);
    cz.wait_front(Dt);
    cw.wait_front(Dt);

    // ss[0,0] = sum_c o[0,c]^2 over the Dt tiles.
    css.reserve_back(1);
    matmul_init(cb_o, cb_o, 1);
    tile_regs_acquire();
    for (uint32_t t = 0; t < Dt; t++) {
        matmul_tiles(cb_o, cb_o, t, t, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_ss);
    tile_regs_release();
    css.push_back(1);
    css.wait_front(1);

    pack_reconfig_data_format(cb_out);  // bf16 output from here on
    cout.reserve_back(Dt);
    for (uint32_t t = 0; t < Dt; t++) {
        tile_regs_acquire();
        copy_init(cb_ss);
        copy_tile(cb_ss, 0, 2);
        binop_with_scalar_tile_init();
        mul_unary_tile(2, inv_d_bits);
        add_unary_tile(2, eps_bits);
        rsqrt_tile_init();
        rsqrt_tile(2);  // DST[2] = rinv (row 0, column 0 is what the broadcast reads)
        copy_init(cb_o);
        copy_tile(cb_o, t, 0);
        if constexpr (stage >= 1) {
            copy_init(cb_w);
            copy_tile(cb_w, t, 1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);  // o * w
        }
        if constexpr (stage >= 2) {
            sfpu_mul_bcast_col_init();
            sfpu_mul_bcast_col(0, 2);  // * rinv, per row from column 0
        }
        if constexpr (stage >= 3) {
            copy_init(cb_z);
            copy_tile(cb_z, t, 1);
            silu_tile_init();
            silu_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0, 1, 0);  // * silu(z)
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, t);
        tile_regs_release();
    }
    cout.push_back(Dt);
    css.pop_front(1);
    co.pop_front(Dt);
    cz.pop_front(Dt);
    cw.pop_front(Dt);
}
