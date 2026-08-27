// SPDX-License-Identifier: Apache-2.0
// Cross-core LayerNorm reduce PROBE compute kernel.
// Isolates the cross-core stats reduction: input x already resides in cb_in as
// this core's N-slice (Ns tiles wide, M_t tiles tall). LayerNorm is over the FULL
// feature dim = num_partials * Ns * 32, spread across the num_partials cores in
// this column. Each core: (1) local partial sum over its Ns tiles -> cb_ex_partial;
// (2) reader all-gathers the num_partials partials into cb_ex_external;
// (3) global mean = sum of partials -> cb_ex; (4) x-mean; (5) local partial sumsq
// -> cb_ex_partial2; (6) gather -> cb_ex_external2; (7) var -> rstd; (8) normalize
// + gamma/beta -> cb_out. Math verbatim from the proven single-core epilogue
// (PCC 0.99998); only the stats source changed local->cross-core.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"

constexpr uint32_t cb_in = tt::CBIndex::c_0;
constexpr uint32_t cb_out = tt::CBIndex::c_2;
constexpr uint32_t cb_gamma = tt::CBIndex::c_5;
constexpr uint32_t cb_beta = tt::CBIndex::c_6;
constexpr uint32_t cb_scaler = tt::CBIndex::c_7;  // 1/N (local partial reduce)
constexpr uint32_t cb_eps = tt::CBIndex::c_8;
constexpr uint32_t cb_ex_partial = tt::CBIndex::c_9;     // local partial sum (M_t)
constexpr uint32_t cb_ex_external = tt::CBIndex::c_10;   // gathered partials (P*M_t)
constexpr uint32_t cb_ex = tt::CBIndex::c_11;            // global mean (M_t)
constexpr uint32_t cb_xmm = tt::CBIndex::c_12;           // x - mean (obn)
constexpr uint32_t cb_ex_partial2 = tt::CBIndex::c_13;   // local partial sumsq (M_t)
constexpr uint32_t cb_ex_external2 = tt::CBIndex::c_14;  // gathered partials2 (P*M_t)
constexpr uint32_t cb_var = tt::CBIndex::c_15;           // global var (M_t)
constexpr uint32_t cb_rstd = tt::CBIndex::c_16;          // rstd (M_t)
constexpr uint32_t cb_xmm2 = tt::CBIndex::c_18;          // xmm^2 (obn)
constexpr uint32_t cb_norm = tt::CBIndex::c_19;          // (x-mean)*rstd (obn)
constexpr uint32_t cb_normg = tt::CBIndex::c_20;         // norm*gamma (obn)
constexpr uint32_t cb_scaler_g = tt::CBIndex::c_17;      // 1.0 (combine reduce)

void kernel_main() {
    constexpr uint32_t M_t = get_compile_time_arg_val(0);
    constexpr uint32_t Ns = get_compile_time_arg_val(1);
    constexpr uint32_t P = get_compile_time_arg_val(2);  // num partials (cores in column)
    constexpr uint32_t obn = M_t * Ns;

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in, cb_scaler, cb_ex_partial);
    cb_wait_front(cb_eps, 1);

    // ---- local partial E[x]: reduce over this core's Ns tiles, scaler=1/N ----
    cb_wait_front(cb_in, obn);
    cb_wait_front(cb_scaler, 1);
    reconfig_data_format(cb_scaler, cb_in);
    reduce_init<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, cb_ex_partial);
    cb_reserve_back(cb_ex_partial, M_t);
    for (uint32_t m = 0; m < M_t; m++) {
        tile_regs_acquire();
        for (uint32_t n = 0; n < Ns; n++) {
            reduce_tile<PoolType::AVG, ReduceDim::REDUCE_ROW>(cb_in, cb_scaler, m * Ns + n, 0, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_ex_partial);
        tile_regs_release();
    }
    reduce_uninit();
    cb_push_back(cb_ex_partial, M_t);

    // ---- global mean = sum of P partials, scaler_global=1.0 ----
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

    // ---- xmm = x - E[x] (bcast col) ----
    cb_wait_front(cb_ex, M_t);
    sub_bcast_cols_init_short(cb_in, cb_ex);
    reconfig_data_format(cb_in, cb_ex);
    pack_reconfig_data_format(cb_xmm);
    cb_reserve_back(cb_xmm, obn);
    for (uint32_t m = 0; m < M_t; m++) {
        for (uint32_t n = 0; n < Ns; n++) {
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_in, cb_ex, m * Ns + n, m, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_xmm);
            tile_regs_release();
        }
    }
    cb_push_back(cb_xmm, obn);
    cb_pop_front(cb_in, obn);

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

    // ---- local partial Var: reduce xmm2 over Ns, scaler=1/N -> cb_ex_partial2 ----
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

    // ---- global var = sum of P partials -> cb_var ----
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
    cb_pop_front(cb_scaler, 1);
}
