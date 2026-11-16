// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B22 fused front norms, compute. P2 = sum_j x_j x_j^T (diag = per-token partial sum of squares of this core's TPC
// tiles; bf16 products exact in fp32); after the exchange rs = rsqrt(R @ (1/W) + eps) (every column = rsqrt of the mean
// square of the token's row); out_j = bf16((x_j * rs) * gamma_j). CT: 0 cb_x 1 cb_g 2 cb_ones 3 cb_p2 4 cb_tot 5 cb_rs
// 6 cb_tmp 7 cb_out 8 TPC 9 EPS_BITS
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

template <VectorMode VM>
ALWI void rsqrt_tile_vm(uint32_t idst) {
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false), idst, VM));
}

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(4);
    constexpr uint32_t cb_rs = get_compile_time_arg_val(5);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t TPC = get_compile_time_arg_val(8);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(9);

    compute_kernel_hw_startup(cb_x, cb_x, cb_p2);
    CircularBuffer x(cb_x), g(cb_g), ones(cb_ones), p2(cb_p2), tot(cb_tot), rs(cb_rs), tmp(cb_tmp), out(cb_out);

    // ---- P2 = sum_j x_j x_j^T ----
    x.wait_front(TPC);
    reconfig_data_format(cb_x, cb_x);
    matmul_init(cb_x, cb_x, 1);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TPC; ++j) {
        matmul_tiles(cb_x, cb_x, j, j, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    p2.reserve_back(1);
    pack_tile(0, cb_p2);
    p2.push_back(1);
    tile_regs_release();

    // ---- rs = rsqrt(R @ (1/W) + eps) ----
    tot.wait_front(1);
    ones.wait_front(1);
    reconfig_data_format(cb_tot, cb_ones);
    matmul_init(cb_tot, cb_ones);
    tile_regs_acquire();
    matmul_tiles(cb_tot, cb_ones, 0, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile_vm<VectorMode::R>(0);
    tile_regs_commit();
    tile_regs_wait();
    rs.reserve_back(1);
    pack_tile(0, cb_rs);
    rs.push_back(1);
    tile_regs_release();

    // ---- tmp_j = x_j * rs ----
    rs.wait_front(1);
    reconfig_data_format(cb_x, cb_rs);
    pack_reconfig_data_format(cb_tmp);
    mul_init(cb_x, cb_rs);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TPC; ++j) {
        mul_tiles(cb_x, cb_rs, j, 0, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    tmp.reserve_back(TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        pack_tile(j, cb_tmp);
    }
    tmp.push_back(TPC);
    tile_regs_release();

    // ---- out_j = bf16(tmp_j * gamma_j) (gamma: row 0 broadcast over the rows) ----
    g.wait_front(TPC);
    tmp.wait_front(TPC);
    reconfig_data_format(cb_tmp, cb_g);
    pack_reconfig_data_format(cb_out);
    mul_bcast_rows_init(cb_tmp, cb_g);
    tile_regs_acquire();
    for (uint32_t j = 0; j < TPC; ++j) {
        mul_tiles_bcast_rows(cb_tmp, cb_g, j, j, j);
    }
    tile_regs_commit();
    tile_regs_wait();
    out.reserve_back(TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        pack_tile(j, cb_out);
    }
    out.push_back(TPC);
    tile_regs_release();
    x.pop_front(TPC);
    g.pop_front(TPC);
    tmp.pop_front(TPC);
    rs.pop_front(1);
    tot.pop_front(1);
    ones.pop_front(1);
}
