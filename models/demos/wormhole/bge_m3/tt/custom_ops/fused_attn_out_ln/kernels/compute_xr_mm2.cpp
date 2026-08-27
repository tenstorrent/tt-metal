// SPDX-License-Identifier: Apache-2.0
// FUSED matmul+residual+cross-core-LN, MULTI M_block streamed (integration step 3).
// Loops MBPC M_blocks; each block runs the proven §14 pipeline. Persistent inputs
// (scaler/scaler_g/gamma/beta/eps) are waited (not popped) every block and popped
// once at the end. Per-block CBs are reused; the reader/writer release protocol
// keeps the external gather buffers reuse-safe.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_interm = tt::CBIndex::c_3;
constexpr uint32_t cb_resid = tt::CBIndex::c_4;
constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
constexpr uint32_t cb_beta = tt::CBIndex::c_6;
constexpr uint32_t cb_scaler = tt::CBIndex::c_7;
constexpr uint32_t cb_eps = tt::CBIndex::c_8;
constexpr uint32_t cb_ex_partial = tt::CBIndex::c_9;
constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;
constexpr uint32_t cb_ex = tt::CBIndex::c_11;
constexpr uint32_t cb_xmm = tt::CBIndex::c_12;
constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_13;
constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_14;
constexpr uint32_t cb_var = tt::CBIndex::c_15;
constexpr uint32_t cb_rstd = tt::CBIndex::c_16;
constexpr uint32_t cb_scaler_g = tt::CBIndex::c_17;
constexpr uint32_t cb_xmm2 = tt::CBIndex::c_18;
constexpr uint32_t cb_norm = tt::CBIndex::c_21;   // ALIAS cb_x (free after xmm) — L1 collapse
constexpr uint32_t cb_normg = tt::CBIndex::c_18;  // ALIAS cb_xmm2 (free after var) — L1 collapse
constexpr uint32_t cb_x = tt::CBIndex::c_21;

#include "compute_matmul_body.hpp"

void kernel_main() {
    constexpr uint32_t M_t = get_compile_time_arg_val(0);
    constexpr uint32_t Ns = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);
    constexpr uint32_t K_block = get_compile_time_arg_val(3);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(6);
    constexpr uint32_t MBPC = get_compile_time_arg_val(7);
    constexpr uint32_t obn = M_t * Ns;
    constexpr uint32_t in0_bt = M_t * K_block;
    constexpr uint32_t in1_bt = K_block * Ns;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_interm);
    matmul_init(cb_in0, cb_in1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t mb = 0; mb < MBPC; mb++) {
        // ---- matmul: accumulate K blocks -> cb_interm ----
        matmul_block_init(cb_in0, cb_in1, false, subblock_w, subblock_h, K_block);
        reconfig_data_format(cb_in1, cb_in0);
        pack_reconfig_data_format(cb_interm);
        cb_reserve_back(cb_interm, obn);
        for (uint32_t k = 0; k < K_num_blocks; k++) {
            cb_wait_front(cb_in0, in0_bt);
            cb_wait_front(cb_in1, in1_bt);
            matmul_blocks(cb_in0, cb_in1, cb_interm, M_t, Ns, Ns, K_block, subblock_h, subblock_w);
            cb_pop_front(cb_in0, in0_bt);
            cb_pop_front(cb_in1, in1_bt);
            if (k == 0) {
                PACK((llk_pack_reconfig_l1_acc(1)));
            }
        }
        cb_push_back(cb_interm, obn);
        PACK((llk_pack_reconfig_l1_acc(0)));

        // ---- x = matmul + residual -> cb_x ----
        cb_wait_front(cb_interm, obn);
        cb_wait_front(cb_resid, obn);
        cb_reserve_back(cb_x, obn);
        add_tiles_init(cb_interm, cb_resid);
        reconfig_data_format(cb_interm, cb_resid);
        pack_reconfig_data_format(cb_x);
        for (uint32_t t = 0; t < obn; t++) {
            tile_regs_acquire();
            add_tiles(cb_interm, cb_resid, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_x);
            tile_regs_release();
        }
        cb_push_back(cb_x, obn);
        cb_pop_front(cb_interm, obn);
        cb_pop_front(cb_resid, obn);

        // ---- local partial E[x], scaler=1/N ----
        cb_wait_front(cb_x, obn);
        cb_wait_front(cb_scaler, 1);
        reconfig_data_format(cb_scaler, cb_x);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, cb_ex_partial);
        cb_reserve_back(cb_ex_partial, M_t);
        for (uint32_t m = 0; m < M_t; m++) {
            tile_regs_acquire();
            for (uint32_t n = 0; n < Ns; n++) {
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, m * Ns + n, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ex_partial);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_ex_partial, M_t);

        // ---- global mean = sum of P partials ----
        cb_wait_front(cb_ex_external, P * M_t);
        cb_wait_front(cb_scaler_g, 1);
        reconfig_data_format(cb_scaler_g, cb_ex_external);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ex_external, cb_scaler_g, cb_ex);
        cb_reserve_back(cb_ex, M_t);
        for (uint32_t m = 0; m < M_t; m++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < P; j++) {
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ex_external, cb_scaler_g, j * M_t + m, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ex);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_ex, M_t);
        cb_pop_front(cb_ex_external, P * M_t);

        // ---- xmm = x - E[x] ----
        cb_wait_front(cb_ex, M_t);
        sub_bcast_cols_init_short(cb_x, cb_ex);
        reconfig_data_format(cb_x, cb_ex);
        pack_reconfig_data_format(cb_xmm);
        cb_reserve_back(cb_xmm, obn);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t n = 0; n < Ns; n++) {
                tile_regs_acquire();
                sub_tiles_bcast_cols(cb_x, cb_ex, m * Ns + n, m, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_xmm);
                tile_regs_release();
            }
        }
        cb_push_back(cb_xmm, obn);
        cb_pop_front(cb_x, obn);

        // ---- xmm2 = xmm^2 ----
        cb_wait_front(cb_xmm, obn);
        mul_tiles_init(cb_xmm, cb_xmm);
        pack_reconfig_data_format(cb_xmm2);
        cb_reserve_back(cb_xmm2, obn);
        for (uint32_t t = 0; t < obn; t++) {
            tile_regs_acquire();
            mul_tiles(cb_xmm, cb_xmm, t, t, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xmm2);
            tile_regs_release();
        }
        cb_push_back(cb_xmm2, obn);

        // ---- local partial Var, scaler=1/N ----
        cb_wait_front(cb_xmm2, obn);
        reconfig_data_format(cb_scaler, cb_xmm2);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_ex_partial2);
        cb_reserve_back(cb_ex_partial2, M_t);
        for (uint32_t m = 0; m < M_t; m++) {
            tile_regs_acquire();
            for (uint32_t n = 0; n < Ns; n++) {
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, m * Ns + n, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_ex_partial2);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_ex_partial2, M_t);
        cb_pop_front(cb_xmm2, obn);

        // ---- global var = sum of P partials ----
        cb_wait_front(cb_ex_external2, P * M_t);
        reconfig_data_format(cb_scaler_g, cb_ex_external2);
        reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ex_external2, cb_scaler_g, cb_var);
        cb_reserve_back(cb_var, M_t);
        for (uint32_t m = 0; m < M_t; m++) {
            tile_regs_acquire();
            for (uint32_t j = 0; j < P; j++) {
                reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_ex_external2, cb_scaler_g, j * M_t + m, 0, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_var);
            tile_regs_release();
        }
        reduce_uninit();
        cb_push_back(cb_var, M_t);
        cb_pop_front(cb_ex_external2, P * M_t);

        // ---- rstd = 1/sqrt(var+eps) ----
        cb_wait_front(cb_var, M_t);
        cb_reserve_back(cb_rstd, M_t);
        add_tiles_init(cb_var, cb_eps);
        for (uint32_t m = 0; m < M_t; m++) {
            tile_regs_acquire();
            add_tiles(cb_var, cb_eps, m, 0, 0);
            rsqrt_tile_init();
            rsqrt_tile(0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_rstd);
            tile_regs_release();
        }
        cb_push_back(cb_rstd, M_t);
        cb_pop_front(cb_var, M_t);

        // ---- normalize + affine ----
        cb_wait_front(cb_xmm, obn);
        cb_wait_front(cb_rstd, M_t);
        cb_wait_front(cb_gamma, Ns);
        cb_wait_front(cb_beta, Ns);
        mul_bcast_cols_init_short(cb_xmm, cb_rstd);
        reconfig_data_format(cb_xmm, cb_rstd);
        pack_reconfig_data_format(cb_norm);
        cb_reserve_back(cb_norm, obn);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t n = 0; n < Ns; n++) {
                tile_regs_acquire();
                mul_tiles_bcast_cols(cb_xmm, cb_rstd, m * Ns + n, m, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_norm);
                tile_regs_release();
            }
        }
        cb_push_back(cb_norm, obn);
        cb_pop_front(cb_xmm, obn);

        cb_wait_front(cb_norm, obn);
        mul_bcast_rows_init_short(cb_norm, cb_gamma);
        reconfig_data_format(cb_norm, cb_gamma);
        pack_reconfig_data_format(cb_normg);
        cb_reserve_back(cb_normg, obn);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t n = 0; n < Ns; n++) {
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_norm, cb_gamma, m * Ns + n, n, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_normg);
                tile_regs_release();
            }
        }
        cb_push_back(cb_normg, obn);
        cb_pop_front(cb_norm, obn);

        cb_wait_front(cb_normg, obn);
        add_bcast_rows_init_short(cb_normg, cb_beta);
        reconfig_data_format(cb_normg, cb_beta);
        pack_reconfig_data_format(cb_out);
        cb_reserve_back(cb_out, obn);
        for (uint32_t m = 0; m < M_t; m++) {
            for (uint32_t n = 0; n < Ns; n++) {
                tile_regs_acquire();
                add_tiles_bcast_rows(cb_normg, cb_beta, m * Ns + n, n, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
            }
        }
        cb_push_back(cb_out, obn);
        cb_pop_front(cb_normg, obn);
        cb_pop_front(cb_ex, M_t);
        cb_pop_front(cb_rstd, M_t);
    }
    cb_pop_front(cb_gamma, Ns);
    cb_pop_front(cb_beta, Ns);
    cb_pop_front(cb_scaler, 1);
    cb_pop_front(cb_scaler_g, 1);
    cb_pop_front(cb_eps, 1);
}
