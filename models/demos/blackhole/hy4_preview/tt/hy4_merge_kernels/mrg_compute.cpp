// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 decode cross-column merge, compute (one core = one user x TPC output tile columns).
//   w_c = l_c exp(ms_c - sink)   (tiles whose column 0 = the value per head row; 8 sources, 2 batches of 4 in DST)
//   den = sum_c w_c + 1 (the sink term) broadcast to all columns by a MAC of ones x w_c (column broadcast); rden = 1/den (fp32 tile)
//   out_j = (sum_c o_{c,j} * w_c  (MAC with column broadcast accumulated in DST, fp32)) * rden
// ms is the scaled max (softmax scale already applied), the sink logit is in the scaled domain: the reference of the softmax is the sink (valid while ms - sink < 88).
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/experimental/eltwise_mul_scalar.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

constexpr uint32_t TPC = get_compile_time_arg_val(0);
constexpr uint32_t CB_ST = get_compile_time_arg_val(1);
constexpr uint32_t CB_A = get_compile_time_arg_val(2);
constexpr uint32_t CB_W = get_compile_time_arg_val(3);
constexpr uint32_t CB_R = get_compile_time_arg_val(4);
constexpr uint32_t CB_ONE = get_compile_time_arg_val(5);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(6);
constexpr uint32_t DBG = get_compile_time_arg_val(7);  // 1: no compute (data movement only), 2: w only
constexpr uint32_t NC = 8;
constexpr uint32_t NB = 4;  // sources per DST batch (fp32 DST half = 4 tiles)

void kernel_main() {
    CircularBuffer st(CB_ST), a(CB_A), w(CB_W), r(CB_R), one(CB_ONE), o(CB_OUT);
    compute_kernel_hw_startup(CB_ST, CB_ONE, CB_OUT);
    st.wait_front(1 + 2 * NC);
    if constexpr (DBG == 1) {
        a.wait_front(NC * TPC);
        o.reserve_back(TPC);
        o.push_back(TPC);
        a.pop_front(NC * TPC);
        st.pop_front(1 + 2 * NC);
        return;
    }
    // ---- w_c = l_c exp(ms_c - sink)
    w.reserve_back(NC);
    for (uint32_t b = 0; b < NC / NB; ++b) {
        reconfig_data_format(CB_ST, CB_ST);
        sub_init(CB_ST, CB_ST);
        tile_regs_acquire();
        for (uint32_t k = 0; k < NB; ++k) {
            sub_tiles(CB_ST, CB_ST, 1 + b * NB + k, 0, k);
        }
        if constexpr (DBG != 3) {
            exp_tile_init();
            for (uint32_t k = 0; k < NB; ++k) {
                exp_tile<false, false, InputClamping::ClampToNegative, 8>(k, VectorMode::R);
            }
        }
        if constexpr (DBG != 3 && DBG != 4) {
            deepseek_binary_dest_reuse_tiles_init(CB_ST);
            for (uint32_t k = 0; k < NB; ++k) {
                deepseek_binary_dest_reuse_tiles(CB_ST, 1 + NC + b * NB + k, k);
            }
        }
        tile_regs_commit();
        pack_reconfig_data_format(CB_W);
        tile_regs_wait();
        for (uint32_t k = 0; k < NB; ++k) {
            pack_tile<true>(k, CB_W, b * NB + k);
        }
        tile_regs_release();
    }
    w.push_back(NC);
    w.wait_front(NC);
    // ---- den = sum_c w_c + 1 (all columns), rden
    one.wait_front(1);
    r.reserve_back(1);
    reconfig_data_format(CB_ONE, CB_W);
    bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(CB_ONE, CB_W);
    MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(CB_ONE, CB_W, 1)));
    tile_regs_acquire();
    for (uint32_t c = 0; c < NC; ++c) {
        mul_tiles_bcast_cols(CB_ONE, CB_W, 0, c, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, 0x3f800000u);  // + 1.0 (the sink term e^0)
    recip_tile_init();
    recip_tile(0, VectorMode::R);
    tile_regs_commit();
    pack_reconfig_data_format(CB_R);
    tile_regs_wait();
    pack_tile(0, CB_R);
    tile_regs_release();
    r.push_back(1);
    r.wait_front(1);
    st.pop_front(1 + 2 * NC);
    if constexpr (DBG == 7 || DBG == 8) {  // debug: output tile j = rden (7) / w_j (8)
        a.wait_front(NC * TPC);
        for (uint32_t j = 0; j < TPC; ++j) {
            o.reserve_back(1);
            tile_regs_acquire();
            if constexpr (DBG == 7) {
                reconfig_data_format_srca(CB_R);
                copy_tile_to_dst_init_short(CB_R);
                copy_tile(CB_R, 0, 0);
            } else {
                reconfig_data_format_srca(CB_W);
                copy_tile_to_dst_init_short(CB_W);
                copy_tile(CB_W, j < NC ? j : 0, 0);
            }
            tile_regs_commit();
            pack_reconfig_data_format(CB_OUT);
            tile_regs_wait();
            pack_tile(0, CB_OUT);
            tile_regs_release();
            o.push_back(1);
        }
        a.pop_front(NC * TPC);
        w.pop_front(NC);
        r.pop_front(1);
        return;
    }
    if constexpr (DBG == 2 || (DBG >= 3 && DBG < 6)) {
        a.wait_front(NC * TPC);
        o.reserve_back(TPC);
        o.push_back(TPC);
        a.pop_front(NC * TPC);
        w.pop_front(NC);
        r.pop_front(1);
        return;
    }
    // ---- out_j = (sum_c o_{c,j} w_c) * rden
    a.wait_front(NC * TPC);
    for (uint32_t j = 0; j < TPC; ++j) {
        reconfig_data_format(CB_A, CB_W);
        bcast_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL>(CB_A, CB_W);
        MATH((llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::COL, MATH_FIDELITY>(CB_A, CB_W, 1)));
        tile_regs_acquire();
        for (uint32_t c = 0; c < NC; ++c) {
            mul_tiles_bcast_cols(CB_A, CB_W, j * NC + c, c, 0);
        }
        if constexpr (DBG != 6) {
            deepseek_binary_dest_reuse_tiles_init(CB_R);
            deepseek_binary_dest_reuse_tiles(CB_R, 0, 0);
        }
        tile_regs_commit();
        o.reserve_back(1);
        pack_reconfig_data_format(CB_OUT);
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
        o.push_back(1);
    }
    a.pop_front(NC * TPC);
    w.pop_front(NC);
    r.pop_front(1);
}
