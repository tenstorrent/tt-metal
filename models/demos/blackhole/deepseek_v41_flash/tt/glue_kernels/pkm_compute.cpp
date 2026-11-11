// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// T19 packed-mHC mixes, compute.  mu = sum_s part_s @ I (fp32 accumulate in DST), ss = mu @ SEL (column NCOL of every
// row broadcast), rs = rsqrt(ss * inv_nc + eps), out = mu * rs.   Same maths as  sum(dim=1) / matmul(sel_ss) / multiply
// / add / rsqrt / multiply.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_part = get_compile_time_arg_val(0);  // UnpackToDestFp32
    constexpr uint32_t cb_eye = get_compile_time_arg_val(1);   // unused (kept for the reader's tile order)
    constexpr uint32_t cb_sel = get_compile_time_arg_val(2);
    constexpr uint32_t cb_mum = get_compile_time_arg_val(3);  // mu for the select matmul (FPU operand)
    constexpr uint32_t cb_rs = get_compile_time_arg_val(4);   // UnpackToDestFp32
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t S = get_compile_time_arg_val(6);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(7);
    constexpr uint32_t INV_BITS = get_compile_time_arg_val(8);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(9);
    constexpr uint32_t cb_mus = get_compile_time_arg_val(10);  // mu for the SFPU multiply, UnpackToDestFp32

    compute_kernel_hw_startup(cb_part, cb_sel, cb_mum);
    CircularBuffer part(cb_part), eye(cb_eye), sel(cb_sel), mum(cb_mum), mus(cb_mus), rs(cb_rs), out(cb_out);
    eye.wait_front(1);
    sel.wait_front(1);

    // mu = sum_s part_s: SFPU adds on full fp32 tiles in DST (copy_tile with unpack-to-dest)
    tile_regs_acquire();
    for (uint32_t s0 = 0; s0 < S; s0 += DEPTH) {
        part.wait_front(DEPTH);
        for (uint32_t i = 0; i < DEPTH; ++i) {
            reconfig_data_format_srca(cb_part);
            copy_tile_to_dst_init_short(cb_part);
            if (s0 == 0 && i == 0) {
                copy_tile(cb_part, i, 0);
            } else {
                copy_tile(cb_part, i, 1);
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
        }
        part.pop_front(DEPTH);
    }
    tile_regs_commit();
    tile_regs_wait();
    mum.reserve_back(1);
    mus.reserve_back(1);
    pack_tile(0, cb_mum);
    pack_tile(0, cb_mus);
    mum.push_back(1);
    mus.push_back(1);
    tile_regs_release();
    mum.wait_front(1);
    mus.wait_front(1);

    // rs = rsqrt(mu[:, NCOL] * inv + eps)
    reconfig_data_format(cb_mum, cb_sel);
    matmul_init(cb_mum, cb_sel);
    tile_regs_acquire();
    matmul_tiles(cb_mum, cb_sel, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rs.wait_front(1);

    // out = mu * rs (SFPU, fp32)
    reconfig_data_format_srca(cb_mus);
    copy_tile_to_dst_init_short(cb_mus);
    tile_regs_acquire();
    copy_tile(cb_mus, 0, 0);
    copy_tile(cb_rs, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(1);
    pack_tile(0, cb_out);
    out.push_back(1);
    tile_regs_release();
}
