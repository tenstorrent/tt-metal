// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Compute for QkvNormRope (one user per core). For each of q (gamma_q, RoPE), k (gamma_k, RoPE), v (no gamma, no
// RoPE): x^2 per column tile on the FPU into fp32 cb_sq; the row sums by an FPU row reduce with the unit scaler;
// inv = rsqrt(sum / D + eps) on the SFPU into fp32 cb_inv; then per rotate_half column pair (a = c, b = c + half):
// n = x * inv (column broadcast) * gamma (row broadcast), out_a = n_a cos_a - n_b sin_a, out_b = n_b cos_b +
// n_a sin_b, all on fp32 dest (8 tiles: full dest sync), packed once to bf16 into the tensor-backed output CB.
// compile-time args: [cb_q, cb_k, cb_v, cb_qo, cb_ko, cb_vo, cb_gq, cb_gk, cb_cos, cb_sin, cb_scaler, cb_sq,
//                     cb_inv, cols_t, half_t, eps_bits, inv_d_bits, cb_n]
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t CB_Q = get_compile_time_arg_val(0);
constexpr uint32_t CB_K = get_compile_time_arg_val(1);
constexpr uint32_t CB_V = get_compile_time_arg_val(2);
constexpr uint32_t CB_QO = get_compile_time_arg_val(3);
constexpr uint32_t CB_KO = get_compile_time_arg_val(4);
constexpr uint32_t CB_VO = get_compile_time_arg_val(5);
constexpr uint32_t CB_GQ = get_compile_time_arg_val(6);
constexpr uint32_t CB_GK = get_compile_time_arg_val(7);
constexpr uint32_t CB_COS = get_compile_time_arg_val(8);
constexpr uint32_t CB_SIN = get_compile_time_arg_val(9);
constexpr uint32_t CB_SCALER = get_compile_time_arg_val(10);
constexpr uint32_t CB_SQ = get_compile_time_arg_val(11);
constexpr uint32_t CB_INV = get_compile_time_arg_val(12);
constexpr uint32_t COLS_T = get_compile_time_arg_val(13);
constexpr uint32_t HALF_T = get_compile_time_arg_val(14);
constexpr uint32_t EPS_BITS = get_compile_time_arg_val(15);
constexpr uint32_t INV_D_BITS = get_compile_time_arg_val(16);
constexpr uint32_t CB_N = get_compile_time_arg_val(17);

// inv column tile of the cb_x shard into cb_inv (fp32): rsqrt(sum(x^2) / D + eps) per row (column 0 valid)
FORCE_INLINE void row_rsqrt(uint32_t cb_x) {
    CircularBuffer csq(CB_SQ), cinv(CB_INV), cs(CB_SCALER);
    // x^2 tiles
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
    // row sums -> column 0, then the SFPU chain on dest
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

// normalize (and rotate) the cb_x shard into cb_out. Dest usage follows the pattern validated by rms_norm_decode:
// every window issues all its unpack copies before its first SFPU op, uses dest tiles 0..3 only, and re-inits the
// SFPU broadcast before each op (an FPU write to a dest tile an in-flight SFPU op reads, and higher dest indices
// under the broadcast helpers, both corrupted q / k in the first version). The normalized columns cross from
// the norm windows to the RoPE windows through the fp32 CB cb_n (cols_t pages).
FORCE_INLINE void norm_rope(uint32_t cb_x, uint32_t cb_g, bool has_gamma, bool rope, uint32_t cb_out) {
    CircularBuffer cinv(CB_INV), cout(cb_out), cg(cb_g), ccos(CB_COS), csin(CB_SIN), cn(CB_N);
    cinv.wait_front(1);
    if (has_gamma) {
        cg.wait_front(COLS_T);
    }
    const uint32_t cb_norm_out = rope ? CB_N : cb_out;
    CircularBuffer cno(cb_norm_out);
    pack_reconfig_data_format(cb_norm_out);
    cno.reserve_back(COLS_T);
    for (uint32_t c = 0; c < COLS_T; ++c) {  // n_c = x_c * inv (column broadcast) * gamma_c (row broadcast)
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
        pack_tile(0, cb_norm_out, c);
        tile_regs_release();
    }
    cno.push_back(COLS_T);
    cinv.pop_front(1);
    if (!rope) {
        return;
    }
    cn.wait_front(COLS_T);
    ccos.wait_front(COLS_T);
    csin.wait_front(COLS_T);
    pack_reconfig_data_format(cb_out);
    cout.reserve_back(COLS_T);
    // Output tiles are packed in page order (c = 0 .. cols_t-1): the packer writes sequentially from the reserved
    // block's start and does not honour an out-of-order pack_tile index (measured: pages 0,4,1,5,... landed in
    // 0,1,2,3,...). Column c < half: out = n_c cos_c - n_{c+half} sin_c; c >= half: out = n_c cos_c + n_{c-half} sin_c.
    for (uint32_t c = 0; c < COLS_T; ++c) {
        const bool first_half = c < HALF_T;
        const uint32_t other = first_half ? c + HALF_T : c - HALF_T;
        tile_regs_acquire();
        reconfig_data_format_srca(CB_N);
        copy_init(CB_N);
        copy_tile(CB_N, c, 0);
        copy_tile(CB_N, other, 2);
        reconfig_data_format_srca(CB_COS);
        copy_init(CB_COS);
        copy_tile(CB_COS, c, 1);
        reconfig_data_format_srca(CB_SIN);
        copy_init(CB_SIN);
        copy_tile(CB_SIN, c, 3);
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);  // n_c cos_c
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(2, 3);  // n_other sin_c
        if (first_half) {
            sub_binary_tile_init();
            sub_binary_tile(0, 2, 0);
        } else {
            add_binary_tile_init();
            add_binary_tile(0, 2, 0);
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
    compute_kernel_hw_startup(CB_Q, CB_SCALER, CB_QO);
    CircularBuffer cq(CB_Q), ck(CB_K), cv(CB_V);
    cq.wait_front(COLS_T);
    ck.wait_front(COLS_T);
    cv.wait_front(COLS_T);
    row_rsqrt(CB_Q);
    norm_rope(CB_Q, CB_GQ, true, true, CB_QO);
    row_rsqrt(CB_K);
    norm_rope(CB_K, CB_GK, true, true, CB_KO);
    row_rsqrt(CB_V);
    norm_rope(CB_V, CB_GQ, false, false, CB_VO);
    cq.pop_front(COLS_T);
    ck.pop_front(COLS_T);
    cv.pop_front(COLS_T);
}
