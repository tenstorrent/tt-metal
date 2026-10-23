// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill per-head RMSNorm (+ gamma on q / k) and q / k RoPE, one work unit = one [32, head_dim] tile row of one
// head (32 consecutive positions). Per unit the reader streams the row's column tiles into CB_X and, for q / k,
// the cos / sin tile rows of the same positions into CB_COS / CB_SIN; this kernel squares every column tile
// (FPU), reduces the row sums (FPU reduce with the unit scaler), takes rsqrt(sum / D + eps) on the SFPU and
// scales each column by it (SFPU column broadcast) and by gamma (row broadcast of the gamma tile's row 0), then
// rotates: out_a = n_a cos_a - n_b sin_b-pair, elementwise over the full tile (every row is its own position, so
// no row broadcast as in the decode op). v: norm only (no gamma, no rope). fp32 dest throughout, one bf16 pack.
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_X = get_compile_time_arg_val(0);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
constexpr uint32_t CB_GQ = get_compile_time_arg_val(2);
constexpr uint32_t CB_GK = get_compile_time_arg_val(3);
constexpr uint32_t CB_COS = get_compile_time_arg_val(4);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(5);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(6);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(7);
constexpr uint32_t CB_INV = get_compile_time_arg_val(8);
constexpr uint32_t CB_N = get_compile_time_arg_val(9);
constexpr uint32_t COLS_T = get_compile_time_arg_val(10);
constexpr uint32_t HALF_T = get_compile_time_arg_val(11);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(12);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(13);

// unit kinds (runtime args): 0 = q (gamma q, rope), 1 = k (gamma k, rope), 2 = v (norm only)
constexpr uint32_t KIND_V = 2;

// rsqrt(mean(x^2) + eps) per row of the CB_X tile row -> CB_INV (fp32, column 0 valid)
FORCE_INLINE void row_rsqrt() {
    CircularBuffer csq(CB_SQ), cinv(CB_INV), cs(CB_SCALER);
    reconfig_data_format(CB_X, CB_X);
    mul_init(CB_X, CB_X);
    pack_reconfig_data_format(CB_SQ);
    csq.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        tile_regs_acquire();
        mul_tiles(CB_X, CB_X, c, c, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SQ, c);
        tile_regs_release();
    }
    csq.push_back(COLS_T);
    csq.wait_front(COLS_T);
    cs.wait_front(1);
    reconfig_data_format(CB_SCALER, CB_SQ);  // REDUCE_ROW SUM: scaler -> srcA, data -> srcB
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, CB_INV);
    tile_regs_acquire();
    for (uint32_t c = 0; c < COLS_T; ++c) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_SQ, CB_SCALER, c, 0, 0);
    }
    reduce_uninit(CB_SQ);
    // sum / D + eps, then rsqrt (SFPU on the fp32 dest tile)
    binop_with_scalar_tile_init();
    mul_unary_tile(0, INV_D_BITS);
    add_unary_tile(0, EPS_BITS);
    rsqrt_tile_init();
    rsqrt_tile<RsqrtMode::Default>(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(CB_INV);
    cinv.reserve_back(1);
    pack_tile(0, CB_INV, 0);
    cinv.push_back(1);
    tile_regs_release();
    csq.pop_front(COLS_T);
}

// CB_X * inv (column broadcast) [* gamma (row broadcast)] -> cb_out (bf16), one window per column tile
FORCE_INLINE void normalize(uint32_t cb_g, bool has_gamma, uint32_t cb_out) {
    CircularBuffer cinv(CB_INV), cout(cb_out), cg(cb_g);
    cinv.wait_front(1);
    if (has_gamma) {
        cg.wait_front(COLS_T);
    }
    pack_reconfig_data_format(cb_out);
    cout.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_X);
        copy_init(CB_X);
        copy_tile(CB_X, c, 0);
        reconfig_data_format_srca(CB_INV);
        copy_init(CB_INV);
        copy_tile(CB_INV, 0, 1);
        if (has_gamma) {
            reconfig_data_format_srca(cb_g);
            copy_init(cb_g);
            copy_tile(cb_g, c, 2);
        }
        sfpu_bcast_col_init();
        sfpu_mul_bcast_col(0, 1);
        if (has_gamma) {
            sfpu_bcast_row_init();
            sfpu_mul_bcast_row(0, 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, c);
        tile_regs_release();
    }
    cout.push_back(COLS_T);
    cinv.pop_front(1);
}

// RoPE of the normalized columns in CB_N (bf16) with the full cos / sin tiles of the unit's positions -> CB_OUT
FORCE_INLINE void rope() {
    CircularBuffer cn(CB_N), ccos(CB_COS), csin(CB_SIN), cout(CB_OUT);
    cn.wait_front(COLS_T);
    ccos.wait_front(COLS_T);
    csin.wait_front(COLS_T);
    pack_reconfig_data_format(CB_OUT);
    cout.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        const bool first_half = c < HALF_T;
        const uint32_t other = first_half ? c + HALF_T : c - HALF_T;
        tile_regs_acquire();
        reconfig_data_format(CB_N, CB_COS);
        mul_init(CB_N, CB_COS);
        mul_tiles(CB_N, CB_COS, c, c, 0);  // n_c * cos_c
        reconfig_data_format(CB_N, CB_SIN);
        mul_init(CB_N, CB_SIN);
        mul_tiles(CB_N, CB_SIN, other, c, 1);  // n_other * sin_c
        if (first_half) {
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT, c);
        tile_regs_release();
    }
    cout.push_back(COLS_T);
    cn.pop_front(COLS_T);
    ccos.pop_front(COLS_T);
    csin.pop_front(COLS_T);
}

void kernel_main() {
    const uint32_t n_units = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT);
    CircularBuffer cx(CB_X);
    for (uint32_t u = 0; u < n_units; ++u) {
        const uint32_t kind = get_arg_val<uint32_t>(1 + u);
        cx.wait_front(COLS_T);
        row_rsqrt();
        if (kind == KIND_V) {
            normalize(CB_GQ, false, CB_OUT);
        } else {
            normalize(kind == 0 ? CB_GQ : CB_GK, true, CB_N);
            rope();
        }
        cx.pop_front(COLS_T);
    }
}
