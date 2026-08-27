// SPDX-License-Identifier: Apache-2.0
// Fused AttnOut matmul + residual + LayerNorm (row over full local N) + affine.
// MILESTONE 1: LOCAL stats over this core's N_block (correct when N_block==full N).
// Clean per-stage CBs (no in-place CB rewrites), mirroring layernorm.cpp dataflow.
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
constexpr uint32_t cb_ex = tt::CBIndex::c_9;
constexpr uint32_t cb_xmm = tt::CBIndex::c_10;
constexpr uint32_t cb_xmm2 = tt::CBIndex::c_11;
constexpr uint32_t cb_var = tt::CBIndex::c_12;
constexpr uint32_t cb_rstd = tt::CBIndex::c_13;
constexpr uint32_t cb_x = tt::CBIndex::c_14;      // x = matmul + residual
constexpr uint32_t cb_norm = tt::CBIndex::c_15;   // (x-mean)*rstd
constexpr uint32_t cb_normg = tt::CBIndex::c_16;  // norm*gamma

#include "compute_matmul_body.hpp"

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);

    constexpr uint32_t in0_bt = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_bt = K_block_tiles * N_block_tiles;
    constexpr uint32_t obn = M_block_tiles * N_block_tiles;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in0, cb_in1, cb_interm);
    matmul_init(cb_in0, cb_in1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t mb = 0; mb < M_blocks_per_core; mb++) {
        for (uint32_t nb = 0; nb < N_blocks_per_core; nb++) {
            // ---- matmul: accumulate K blocks -> cb_interm ----
            matmul_block_init(cb_in0, cb_in1, false, subblock_w, subblock_h, K_block_tiles);
            reconfig_data_format(cb_in1, cb_in0);
            pack_reconfig_data_format(cb_interm);
            cb_reserve_back(cb_interm, obn);
            for (uint32_t k = 0; k < K_num_blocks; k++) {
                cb_wait_front(cb_in0, in0_bt);
                cb_wait_front(cb_in1, in1_bt);
                matmul_blocks(
                    cb_in0,
                    cb_in1,
                    cb_interm,
                    M_block_tiles,
                    N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    subblock_h,
                    subblock_w);
                cb_pop_front(cb_in0, in0_bt);
                cb_pop_front(cb_in1, in1_bt);
                if (k == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            cb_push_back(cb_interm, obn);
            PACK((llk_pack_reconfig_l1_acc(0)));

            // ---- x = matmul + residual  -> cb_x ----
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
#ifdef BYPASS_LN
            cb_wait_front(cb_x, obn);
            copy_tile_to_dst_init_short(cb_x);
            reconfig_data_format_srca(cb_x);
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, obn);
            for (uint32_t t = 0; t < obn; t++) {
                tile_regs_acquire();
                copy_tile(cb_x, t, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
            }
            cb_push_back(cb_out, obn);
            cb_pop_front(cb_x, obn);
            continue;
#endif

            // ---- E[x] per row -> cb_ex ----
            cb_wait_front(cb_x, obn);
            cb_wait_front(cb_scaler, 1);
            reconfig_data_format(cb_scaler, cb_x);
            reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, cb_ex);
            cb_reserve_back(cb_ex, M_block_tiles);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                tile_regs_acquire();
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, m * N_block_tiles + n, 0, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_ex);
                tile_regs_release();
            }
            reduce_uninit();
            cb_push_back(cb_ex, M_block_tiles);
#ifdef DUMP_MEAN
            cb_wait_front(cb_ex, M_block_tiles);
            // broadcast mean(col0) across all columns -> cb_out (so torch can read per-row mean)
            sub_bcast_cols_init_short(cb_x, cb_ex);  // reuse: 0 - (-mean)? no; use add with zero
            // Simple: copy mean tile broadcast by adding to zero via bcast is complex; just pack mean tile raw.
            copy_tile_to_dst_init_short(cb_ex);
            reconfig_data_format_srca(cb_ex);
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, obn);
            for (uint32_t t = 0; t < obn; t++) {
                tile_regs_acquire();
                copy_tile(cb_ex, 0, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_out);
                tile_regs_release();
            }
            cb_push_back(cb_out, obn);
            cb_pop_front(cb_ex, M_block_tiles);
            cb_pop_front(cb_x, obn);
            continue;
#endif

            // ---- xmm = x - E[x] ----
            cb_wait_front(cb_ex, M_block_tiles);
            sub_bcast_cols_init_short(cb_x, cb_ex);
            reconfig_data_format(cb_x, cb_ex);
            pack_reconfig_data_format(cb_xmm);
            cb_reserve_back(cb_xmm, obn);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    tile_regs_acquire();
                    sub_tiles_bcast_cols(cb_x, cb_ex, m * N_block_tiles + n, m, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_xmm);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_xmm, obn);
            cb_pop_front(cb_x, obn);

            // ---- xmm2 = xmm^2 ; Var = mean(xmm2) ----
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

            cb_wait_front(cb_xmm2, obn);
            reconfig_data_format(cb_scaler, cb_xmm2);
            reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, cb_var);
            cb_reserve_back(cb_var, M_block_tiles);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                tile_regs_acquire();
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_xmm2, cb_scaler, m * N_block_tiles + n, 0, 0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_var);
                tile_regs_release();
            }
            reduce_uninit();
            cb_push_back(cb_var, M_block_tiles);
            cb_pop_front(cb_xmm2, obn);

            // ---- rstd = 1/sqrt(var+eps) ----
            cb_wait_front(cb_var, M_block_tiles);
            cb_reserve_back(cb_rstd, M_block_tiles);
            add_tiles_init(cb_var, cb_eps);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                tile_regs_acquire();
                add_tiles(cb_var, cb_eps, m, 0, 0);
                rsqrt_tile_init();
                rsqrt_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_rstd);
                tile_regs_release();
            }
            cb_push_back(cb_rstd, M_block_tiles);
            cb_pop_front(cb_var, M_block_tiles);

            // ---- normalize + affine (clean, no in-place) ----
            cb_wait_front(cb_xmm, obn);
            cb_wait_front(cb_rstd, M_block_tiles);
            cb_wait_front(cb_gamma, N_block_tiles);
            cb_wait_front(cb_beta, N_block_tiles);
            // 6a: norm = (x-mean) * rstd  (bcast col) -> cb_norm
            mul_bcast_cols_init_short(cb_xmm, cb_rstd);
            reconfig_data_format(cb_xmm, cb_rstd);
            pack_reconfig_data_format(cb_norm);
            cb_reserve_back(cb_norm, obn);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_cols(cb_xmm, cb_rstd, m * N_block_tiles + n, m, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_norm);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_norm, obn);
            cb_pop_front(cb_xmm, obn);
            // 6b: normg = norm * gamma (bcast row) -> cb_normg
            cb_wait_front(cb_norm, obn);
            mul_bcast_rows_init_short(cb_norm, cb_gamma);
            reconfig_data_format(cb_norm, cb_gamma);
            pack_reconfig_data_format(cb_normg);
            cb_reserve_back(cb_normg, obn);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    tile_regs_acquire();
                    mul_tiles_bcast_rows(cb_norm, cb_gamma, m * N_block_tiles + n, n, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_normg);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_normg, obn);
            cb_pop_front(cb_norm, obn);
            // 6c: out = normg + beta (bcast row) -> cb_out
            cb_wait_front(cb_normg, obn);
            add_bcast_rows_init_short(cb_normg, cb_beta);
            reconfig_data_format(cb_normg, cb_beta);
            pack_reconfig_data_format(cb_out);
            cb_reserve_back(cb_out, obn);
            for (uint32_t m = 0; m < M_block_tiles; m++) {
                for (uint32_t n = 0; n < N_block_tiles; n++) {
                    tile_regs_acquire();
                    add_tiles_bcast_rows(cb_normg, cb_beta, m * N_block_tiles + n, n, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_out);
                    tile_regs_release();
                }
            }
            cb_push_back(cb_out, obn);
            cb_pop_front(cb_normg, obn);
            cb_pop_front(cb_ex, M_block_tiles);
            cb_pop_front(cb_rstd, M_block_tiles);
        }
    }
    cb_pop_front(cb_scaler, 1);
}
