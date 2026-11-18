// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill merged PRE program, compute (one core per token tile-row, whole row on the core): phases A (stats: M,
// Q) B (pp) C (collapse + sum of squares of h, h kept in L1) D (rms norm, bf16 out), the code of the 4-program version
// (K1..K4) back to back. cb_xc, cb_s, cb_acc, cb_h, cb_rs, cb_m, cb_cst2 are unpacked straight to dest (fp32 exact);
// cb_x (matmul path) is separate from cb_xc. CT: 0 cb_x, 1 cb_w, 2 cb_cst, 3 cb_cst2, 4 cb_m, 5 cb_q, 6 cb_pp, 7 cb_s,
// 8 cb_xc, 9 cb_h, 10 cb_acc, 11 cb_asum, 12 cb_rs, 13 cb_wn, 14 cb_out, 15 NTD, 16 INV_N bits, 17 EPSB bits, 18 INV_D
// bits.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_cst = get_compile_time_arg_val(2);
    constexpr uint32_t cb_cst2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_m = get_compile_time_arg_val(4);
    constexpr uint32_t cb_q = get_compile_time_arg_val(5);
    constexpr uint32_t cb_pp = get_compile_time_arg_val(6);
    constexpr uint32_t cb_s = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xc = get_compile_time_arg_val(8);
    constexpr uint32_t cb_h = get_compile_time_arg_val(9);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(10);
    constexpr uint32_t cb_asum = get_compile_time_arg_val(11);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(12);
    constexpr uint32_t cb_wn = get_compile_time_arg_val(13);
    constexpr uint32_t cb_out = get_compile_time_arg_val(14);
    constexpr uint32_t NTD = get_compile_time_arg_val(15);
    constexpr uint32_t INV_N = get_compile_time_arg_val(16);
    constexpr uint32_t EPSB = get_compile_time_arg_val(17);
    constexpr uint32_t INV_D = get_compile_time_arg_val(18);
    constexpr uint32_t B = 4;
    compute_kernel_hw_startup(cb_x, cb_w, cb_out);
    CircularBuffer x(cb_x), w(cb_w), cst(cb_cst), cst2(cb_cst2), m(cb_m), q(cb_q), pp(cb_pp), s(cb_s), xc(cb_xc),
        h(cb_h), acc(cb_acc), asum(cb_asum), rs(cb_rs), wn(cb_wn), out(cb_out);
    cst.wait_front(1);
    cst2.wait_front(4);
    // ---- A: M = sum x @ W (dest 0), Q = sum x^2 (dest 1)
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    fill_tile(1, 0.0f);
    for (uint32_t b = 0; b < 4 * NTD / B; ++b) {
        x.wait_front(B);
        w.wait_front(B);
        reconfig_data_format(cb_w, cb_x);
        matmul_init(cb_x, cb_w);
        for (uint32_t i = 0; i < B; ++i) {
            matmul_tiles(cb_x, cb_w, i, i, 0);
        }
        reconfig_data_format_srca(cb_x);
        copy_tile_to_dst_init_short(cb_x);
        for (uint32_t i = 0; i < B; ++i) {
            copy_tile(cb_x, i, 2 + i);
        }
        square_tile_init();
        for (uint32_t i = 0; i < B; ++i) {
            square_tile(2 + i);
        }
        add_binary_tile_init();
        for (uint32_t i = 0; i < B; ++i) {
            add_binary_tile(1, 2 + i, 1);
        }
        x.pop_front(B);
        w.pop_front(B);
    }
    tile_regs_commit();
    tile_regs_wait();
    m.reserve_back(1);
    q.reserve_back(1);
    pack_reconfig_data_format(cb_m);
    pack_tile(0, cb_m);
    pack_tile(1, cb_q);
    m.push_back(1);
    q.push_back(1);
    tile_regs_release();
    // ---- B: pp
    m.wait_front(1);
    q.wait_front(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_q, cb_cst);
    matmul_init(cb_q, cb_cst);
    matmul_tiles(cb_q, cb_cst, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_N);
    add_unary_tile(0, EPSB);
    rsqrt_tile_init();
    rsqrt_tile(0);
    reconfig_data_format_srca(cb_m);
    copy_tile_to_dst_init_short(cb_m);
    copy_tile(cb_m, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    reconfig_data_format_srca(cb_cst2);
    copy_tile_to_dst_init_short(cb_cst2);
    copy_tile(cb_cst2, 0, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    copy_tile(cb_cst2, 1, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    copy_tile_to_dst_init_short(cb_cst2);
    copy_tile(cb_cst2, 2, 1);
    mul_binary_tile_init();
    mul_binary_tile(0, 1, 0);
    copy_tile(cb_cst2, 3, 1);
    add_binary_tile_init();
    add_binary_tile(0, 1, 0);
    tile_regs_commit();
    tile_regs_wait();
    pp.reserve_back(1);
    pack_reconfig_data_format(cb_pp);
    pack_tile(0, cb_pp);
    pp.push_back(1);
    tile_regs_release();
    // ---- C: collapse + sum of squares of h
    s.wait_front(4);
    for (uint32_t j = 0; j < NTD; ++j) {
        if ((j & 1) == 0) {
            xc.wait_front(8);
        }
        const uint32_t xo = (j & 1) * 4;
        if (j > 0) {
            acc.wait_front(1);
        }
        tile_regs_acquire();
        reconfig_data_format_srca(cb_xc);
        copy_tile_to_dst_init_short(cb_xc);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_xc, xo + k, k);
        }
        reconfig_data_format_srca(cb_s);
        copy_tile_to_dst_init_short(cb_s);
        for (uint32_t k = 0; k < 4; ++k) {
            copy_tile(cb_s, k, 4 + k);
        }
        mul_binary_tile_init();
        for (uint32_t k = 0; k < 4; ++k) {
            mul_binary_tile(k, 4 + k, k);
        }
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
        add_binary_tile(0, 2, 0);
        add_binary_tile(0, 3, 0);
        mul_binary_tile_init();
        mul_binary_tile(0, 0, 1);
        if (j > 0) {
            reconfig_data_format_srca(cb_acc);
            copy_tile_to_dst_init_short(cb_acc);
            copy_tile(cb_acc, 0, 2);
            add_binary_tile_init();
            add_binary_tile(1, 2, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        if (j > 0) {
            acc.pop_front(1);
        }
        h.reserve_back(1);
        acc.reserve_back(1);
        pack_reconfig_data_format(cb_h);
        pack_tile(0, cb_h);
        pack_tile(1, cb_acc);
        h.push_back(1);
        acc.push_back(1);
        tile_regs_release();
        if ((j & 1) == 1) {
            xc.pop_front(8);
        }
    }
    // ---- D: rs, out_j = bf16((h_j * rs) * wn_j)
    acc.wait_front(1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_acc);
    copy_tile_to_dst_init_short(cb_acc);
    copy_tile(cb_acc, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    asum.reserve_back(1);
    pack_reconfig_data_format(cb_asum);
    pack_tile(0, cb_asum);
    asum.push_back(1);
    tile_regs_release();
    asum.wait_front(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_asum, cb_cst);
    matmul_init(cb_asum, cb_cst);
    matmul_tiles(cb_asum, cb_cst, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_D);
    add_unary_tile(0, EPSB);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_reconfig_data_format(cb_rs);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rs.wait_front(1);
    for (uint32_t j = 0; j < NTD; j += 2) {
        h.wait_front(2);
        wn.wait_front(2);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_h);
        copy_tile_to_dst_init_short(cb_h);
        copy_tile(cb_h, 0, 0);
        copy_tile(cb_h, 1, 1);
        reconfig_data_format_srca(cb_wn);
        copy_tile_to_dst_init_short(cb_wn);
        copy_tile(cb_wn, 0, 2);
        copy_tile(cb_wn, 1, 3);
        reconfig_data_format_srca(cb_rs);
        copy_tile_to_dst_init_short(cb_rs);
        copy_tile(cb_rs, 0, 7);
        mul_binary_tile_init();
        mul_binary_tile(0, 7, 0);
        mul_binary_tile(1, 7, 1);
        mul_binary_tile(0, 2, 0);
        mul_binary_tile(1, 3, 1);
        tile_regs_commit();
        tile_regs_wait();
        out.reserve_back(2);
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        pack_tile(1, cb_out);
        out.push_back(2);
        tile_regs_release();
        h.pop_front(2);
        wn.pop_front(2);
    }
}
