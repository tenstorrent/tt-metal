// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// B49 iHC prefill K4 compute: RS = rsqrt(sum_cols(sum_k A_k) / D + eps) (all columns), out_j = bf16((h_j * RS) * wn_j)
// (exact fp32 SFPU products, pairs of tiles per section). CT: 0 cb_c, 1 cb_a, 2 cb_h, 3 cb_w, 4 cb_asum, 5 cb_rs, 6
// cb_out, 7 CT, 8 NCH, 9 INV_D bits, 10 EPSB bits. cb_a, cb_h, cb_w, cb_rs are unpacked straight to dest.
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
    constexpr uint32_t cb_c = get_compile_time_arg_val(0);
    constexpr uint32_t cb_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_h = get_compile_time_arg_val(2);
    constexpr uint32_t cb_w = get_compile_time_arg_val(3);
    constexpr uint32_t cb_asum = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out = get_compile_time_arg_val(6);
    constexpr uint32_t CT = get_compile_time_arg_val(7);
    constexpr uint32_t NCH = get_compile_time_arg_val(8);
    constexpr uint32_t INV_D = get_compile_time_arg_val(9);
    constexpr uint32_t EPSB = get_compile_time_arg_val(10);
    compute_kernel_hw_startup(cb_h, cb_w, cb_out);
    CircularBuffer c(cb_c), a(cb_a), h(cb_h), w(cb_w), asum(cb_asum), rs(cb_rs), out(cb_out);
    c.wait_front(1);
    a.wait_front(NCH);
    // asum = sum_k A_k (exact)
    tile_regs_acquire();
    reconfig_data_format_srca(cb_a);
    copy_tile_to_dst_init_short(cb_a);
    copy_tile(cb_a, 0, 0);
    for (uint32_t k = 1; k < NCH; ++k) {
        copy_tile_to_dst_init_short(cb_a);
        copy_tile(cb_a, k, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    asum.reserve_back(1);
    pack_reconfig_data_format(cb_asum);
    pack_tile(0, cb_asum);
    asum.push_back(1);
    tile_regs_release();
    // rs
    asum.wait_front(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0f);
    reconfig_data_format(cb_asum, cb_c);
    matmul_init(cb_asum, cb_c);
    matmul_tiles(cb_asum, cb_c, 0, 0, 0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_D);
    add_unary_tile(0, EPSB);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();
    rs.wait_front(1);
    for (uint32_t j = 0; j < CT; j += 2) {
        h.wait_front(2);
        w.wait_front(2);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_h);
        copy_tile_to_dst_init_short(cb_h);
        copy_tile(cb_h, 0, 0);
        copy_tile(cb_h, 1, 1);
        reconfig_data_format_srca(cb_w);
        copy_tile_to_dst_init_short(cb_w);
        copy_tile(cb_w, 0, 2);
        copy_tile(cb_w, 1, 3);
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
        w.pop_front(2);
    }
}
