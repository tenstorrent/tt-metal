// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for QkvHeadsNormRope: the norm + RoPE stages of custom_ops/qkv_norm_rope's v2 compute (x^2 on the FPU,
// FPU row reduce with the unit scaler, rsqrt on the SFPU, x * inv (SFPU column broadcast) * gamma (SFPU row
// broadcast) into the bf16 CB cb_n, then rope_decode's FPU form: out_c = n_c cos_c - n_{c+half} sin_c /
// out_c = n_c cos_c + n_{c-half} sin_c with cos / sin row-broadcast), for ONE tensor per core: q or k (gamma +
// RoPE) or v (norm only), selected by the runtime arg; the output goes to cb_out (the writer ships it to the
// user's shard).
// compile-time args: [cb_x, cb_out, cb_g, cb_cos, cb_sin, cb_scaler, cb_sq, cb_inv, cols_t, half_t, eps_bits,
//                     inv_d_bits, cb_n]
// runtime args: [tensor (0 q, 1 k, 2 v)]
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
constexpr uint32_t CB_G = get_compile_time_arg_val(2);
constexpr uint32_t CB_COS = get_compile_time_arg_val(3);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(4);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(5);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(6);
constexpr uint32_t CB_INV = get_compile_time_arg_val(7);
constexpr uint32_t COLS_T = get_compile_time_arg_val(8);
constexpr uint32_t HALF_T = get_compile_time_arg_val(9);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(10);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(11);
constexpr uint32_t CB_N = get_compile_time_arg_val(12);

// inv column tile of the cb_x shard into cb_inv (fp32): rsqrt(sum(x^2) / D + eps) per row (column 0 valid)
FORCE_INLINE void row_rsqrt(uint32_t cb_x) {
    CircularBuffer csq(CB_SQ), cinv(CB_INV), cs(CB_SCALER);
    reconfig_data_format(cb_x, cb_x);
    mul_init(cb_x, cb_x);
    pack_reconfig_data_format(CB_SQ);
    csq.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, c, c, 0);
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

// cb_x * inv (column broadcast) [* gamma (row broadcast)] -> cb_out (bf16), one window per column tile, page order
FORCE_INLINE void normalize(uint32_t cb_x, uint32_t cb_g, bool has_gamma, uint32_t cb_out) {
    CircularBuffer cinv(CB_INV), cout(cb_out), cg(cb_g);
    cinv.wait_front(1);
    if (has_gamma) {
        cg.wait_front(COLS_T);
    }
    pack_reconfig_data_format(cb_out);
    cout.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        tile_regs_acquire();
        reconfig_data_format_srca(cb_x);
        copy_init(cb_x);
        copy_tile(cb_x, c, 0);
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

// RoPE of the normalized columns in cb_n (bf16) into cb_out, one output column per window, page order
FORCE_INLINE void rope(uint32_t cb_out) {
    CircularBuffer cn(CB_N), ccos(CB_COS), csin(CB_SIN), cout(cb_out);
    cn.wait_front(COLS_T);
    ccos.wait_front(COLS_T);
    csin.wait_front(COLS_T);
    pack_reconfig_data_format(cb_out);
    cout.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {
        const bool first_half = c < HALF_T;
        const uint32_t other = first_half ? c + HALF_T : c - HALF_T;
        tile_regs_acquire();
        reconfig_data_format(CB_N, CB_COS);
        mul_bcast_rows_init(CB_N, CB_COS);
        mul_tiles_bcast_rows(CB_N, CB_COS, c, c, 0);  // n_c * cos_c
        reconfig_data_format(CB_N, CB_SIN);
        mul_bcast_rows_init(CB_N, CB_SIN);
        mul_tiles_bcast_rows(CB_N, CB_SIN, other, c, 1);  // n_other * sin_c
        if (first_half) {
            sub_binary_tile_init();
            sub_binary_tile(0, 1, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out, c);
        tile_regs_release();
    }
    cout.push_back(COLS_T);
    cn.pop_front(COLS_T);
}

void kernel_main() {
    const uint32_t tensor = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_X, CB_SCALER, CB_OUT);
    CircularBuffer cx(CB_X);
    cx.wait_front(COLS_T);
    row_rsqrt(CB_X);
    if (tensor < 2) {
        normalize(CB_X, CB_G, true, CB_N);
        rope(CB_OUT);
    } else {
        normalize(CB_X, CB_G, false, CB_OUT);
    }
    cx.pop_front(COLS_T);
}
