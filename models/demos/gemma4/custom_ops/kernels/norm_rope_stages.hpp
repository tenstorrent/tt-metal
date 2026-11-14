// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// The per-head RMSNorm and RoPE stages shared by the fused q / k / v compute kernels (custom_ops/qkv_norm_rope,
// qkv_heads_norm_rope, qkv_norm_rope_prefill). A stage works on one tile row of COLS_T column tiles streamed
// through circular buffers, fp32 dest throughout, one bf16 pack per column tile in page order:
//   row_rsqrt: x^2 on the FPU (one window per column tile) -> CB_SQ, the row sums by an FPU row reduce with the
//              unit scaler of CB_SCALER, sum / D + eps and rsqrt on the SFPU -> CB_INV (fp32, column 0 valid);
//   normalize: x * inv (SFPU column broadcast of CB_INV) [* gamma (SFPU row broadcast of row 0 of the gamma
//              tile)] -> the output CB (bf16); pops CB_INV;
//   rope:      out_c = n_c cos_c - n_{c + half} sin_c (c < HALF_T), n_c cos_c + n_{c - half} sin_c (c >= HALF_T)
//              from the normalized columns in CB_N and the cos / sin tiles of CB_COS / CB_SIN, the two products
//              combined on fp32 dest by the binary SFPU; ROW_BCAST = the decode form (the user's cos / sin row in
//              row 0 of the tile, FPU row broadcast), else elementwise over the full tiles (prefill: every row is
//              its own position). Pops CB_N; the caller pops cos / sin when it refills them.
// Every function is FORCE_INLINE and templated on the CB indices and tile constants, so each kernel's
// compile-time args bind them without a runtime table.
#pragma once
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

namespace norm_rope {

// inv column tile of the cb_x tile row into CB_INV (fp32): rsqrt(sum(x^2) * inv_d + eps) per row (column 0 valid)
template <uint32_t CB_SQ, uint32_t CB_INV, uint32_t CB_SCALER, uint32_t COLS_T, uint32_t INV_D_BITS, uint32_t EPS_BITS>
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
template <uint32_t CB_INV, uint32_t COLS_T>
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

// RoPE of the normalized columns in CB_N (bf16) with the cos / sin tiles -> cb_out, one output column per window
template <uint32_t CB_N, uint32_t CB_COS, uint32_t CB_SIN, uint32_t COLS_T, uint32_t HALF_T, bool ROW_BCAST>
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
        if constexpr (ROW_BCAST) {
            mul_bcast_rows_init(CB_N, CB_COS);
            mul_tiles_bcast_rows(CB_N, CB_COS, c, c, 0);  // n_c * cos_c
        } else {
            mul_init(CB_N, CB_COS);
            mul_tiles(CB_N, CB_COS, c, c, 0);
        }
        reconfig_data_format(CB_N, CB_SIN);
        if constexpr (ROW_BCAST) {
            mul_bcast_rows_init(CB_N, CB_SIN);
            mul_tiles_bcast_rows(CB_N, CB_SIN, other, c, 1);  // n_other * sin_c
        } else {
            mul_init(CB_N, CB_SIN);
            mul_tiles(CB_N, CB_SIN, other, c, 1);
        }
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

}  // namespace norm_rope
