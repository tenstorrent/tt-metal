// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(7) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(8) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(9) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(10) == 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr auto cb_x = tt::CB::c_in0;       // input
    constexpr auto cb_scaler = tt::CB::c_in1;  // scaler
    constexpr auto cb_eps = tt::CB::c_in2;     // epsilon
    constexpr auto cb_gamma = tt::CB::c_in3;   // gamma
    constexpr auto cb_beta = tt::CB::c_in4;    // beta
    constexpr auto cb_mask_h = tt::CB::c_in5;  // mask_h
    constexpr auto cb_mask_w = tt::CB::c_in6;  // mask_w

    constexpr auto cb_out = tt::CB::c_out0;   // output
    constexpr auto cb_mean = tt::CB::c_out1;  // mean
    constexpr auto cb_rstd = tt::CB::c_out2;  // rstd

    constexpr auto cb_ex = tt::CB::c_intermed0;          // E[x]
    constexpr auto cb_xmm = tt::CB::c_intermed1;         // x - E[x]
    constexpr auto cb_xmm2 = tt::CB::c_intermed2;        // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CB::c_intermed3;     // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CB::c_intermed4;         // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_rstd = tt::CB::c_intermed5;  // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CB::c_intermed6;  // p * gamm + beta
    constexpr auto cb_xsum = tt::CB::c_intermed7;        // Sum[x]

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader
    cb_wait_front(cb_eps, onetile);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    constexpr uint32_t NCHt = num_rows_per_core;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t first_tile = 0;

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        /*
         * Sum[x]
         * cb_xsum
         */
        cb_wait_front(cb_x, Wt);
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = wt + j;
                if (w_idx == 0) {
                    ACQ();
                    cb_reserve_back(cb_xsum, onetile);

                    copy_tile_init();
                    copy_tile(cb_x, first_tile, dst0);  // input

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile(cb_mask_h, first_tile, dst1);  // mask_h
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile(cb_mask_w, first_tile, dst1);  // mask_w
                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }

                    pack_tile(dst0, cb_xsum);
                    cb_push_back(cb_xsum, onetile);
                    REL();
                } else {
                    ACQ();
                    // I use cb_ex temporarily.
                    constexpr auto cb_tmp = cb_ex;
                    cb_reserve_back(cb_tmp, onetile);

                    copy_tile_init();
                    copy_tile(cb_x, wt + j, dst0);  // input

                    const uint32_t mask_dst = dst0 < 15 ? dst0 + 1 : 0;

                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile_init();
                        mask_tile(dst0, mask_dst);
                    }

                    if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile_init();
                        mask_tile(dst0, mask_dst);
                    }

                    pack_tile(dst0, cb_tmp);
                    cb_push_back(cb_tmp, onetile);
                    REL();

                    ACQ();
                    cb_wait_front(cb_tmp, onetile);
                    cb_wait_front(cb_xsum, onetile);
                    cb_reserve_back(cb_xsum, onetile);

                    add_tiles_init();
                    add_tiles(cb_xsum, cb_tmp, first_tile, first_tile, dst0);
                    pack_tile(dst0, cb_xsum);

                    cb_pop_front(cb_tmp, onetile);
                    cb_pop_front(cb_xsum, onetile);
                    cb_push_back(cb_xsum, onetile);
                    REL();
                }
            }  // block_size loop
        }      // Wt loop
        // We don't pop cb_x until we compute xmm.

        /*
         * E[x]
         * cb_ex
         */
        ACQ();
        cb_wait_front(cb_xsum, onetile);
        cb_reserve_back(cb_ex, onetile);

        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xsum, cb_scaler, first_tile, first_tile, dst0);
        reduce_revert_delta();

        pack_tile(dst0, cb_ex);

        cb_pop_front(cb_xsum, onetile);
        cb_push_back(cb_ex, onetile);
        REL();

        cb_wait_front(cb_ex, onetile);
        if (mean_has_value) {
            // Write on cb_mean.
            ACQ();
            cb_reserve_back(cb_mean, onetile);

            copy_tile_init();
            copy_tile(cb_ex, first_tile, dst0);

            pack_tile(dst0, cb_mean);

            cb_push_back(cb_mean, onetile);
            REL();
        }
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        cb_reserve_back(cb_xmm, Wt);
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = wt + j;
                ACQ();
                if (is_lastdim_layernorm) {
                    sub_bcast_cols_init_short();
                    sub_tiles_bcast_cols(cb_x, cb_ex, w_idx, first_tile, j);
                } else {
                    sub_tiles_bcast_scalar_init_short();
                    sub_tiles_bcast_scalar(cb_x, cb_ex, w_idx, first_tile, j);
                }
                // mask xmm
                if (do_mask_h || do_mask_w) {
                    const uint32_t mask_dst = j < 15 ? j + 1 : 0;
                    copy_tile_init();
                    mask_tile_init();
                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile(cb_mask_h, first_tile, mask_dst);  // mask_h
                        mask_tile(j, mask_dst);
                    }
                    if (do_mask_w && (w_idx + 1) % origin_Wt == 0) {
                        copy_tile(cb_mask_w, first_tile, mask_dst);  // mask_w
                        mask_tile(j, mask_dst);
                    }
                }
                pack_tile(j, cb_xmm);
                REL();
            }  // block_size loop
            cb_push_back(cb_xmm, block_size);
        }  // Wt loop
        cb_pop_front(cb_ex, onetile);
        cb_pop_front(cb_x, Wt);

        /*
         * Sum[(x - E[x])^2]
         * cb_xmm2sum
         */
        cb_wait_front(cb_xmm, Wt);
        for (uint32_t wt = 0; wt < Wt; wt++) {
            ACQ();
            cb_reserve_back(cb_xmm2, onetile);

            mul_tiles_init();
            mul_tiles(cb_xmm, cb_xmm, wt, wt, dst0);
            pack_tile(dst0, cb_xmm2);

            cb_push_back(cb_xmm2, onetile);
            REL();
            if (wt == 0) {
                ACQ();
                cb_wait_front(cb_xmm2, onetile);
                cb_reserve_back(cb_xmm2sum, onetile);

                copy_tile_init();
                copy_tile(cb_xmm2, first_tile, dst0);
                pack_tile(dst0, cb_xmm2sum);

                cb_pop_front(cb_xmm2, onetile);
                cb_push_back(cb_xmm2sum, onetile);
                REL();
            } else {
                ACQ();
                cb_wait_front(cb_xmm2sum, onetile);
                cb_wait_front(cb_xmm2, onetile);
                cb_reserve_back(cb_xmm2sum, onetile);

                add_tiles_init();
                add_tiles(cb_xmm2sum, cb_xmm2, first_tile, first_tile, dst0);
                pack_tile(dst0, cb_xmm2sum);

                cb_pop_front(cb_xmm2sum, onetile);
                cb_pop_front(cb_xmm2, onetile);
                cb_push_back(cb_xmm2sum, onetile);
                REL();
            }
        }  // Wt loop
        // We don't pop cb_xmm here.

        /*
         * E[(x-E[x])^2 = Var[x]
         * cb_var
         */
        ACQ();
        cb_wait_front(cb_xmm2sum, onetile);
        cb_reserve_back(cb_var, onetile);

        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        reduce_tile(REDUCE_OP, REDUCE_DIM, cb_xmm2sum, cb_scaler, first_tile, first_tile, dst0);
        reduce_revert_delta();

        pack_tile(dst0, cb_var);

        cb_pop_front(cb_xmm2sum, onetile);
        cb_push_back(cb_var, onetile);
        REL();

        cb_wait_front(cb_var, onetile);
        if (rstd_has_value) {
            // Write on cb_rstd.
            ACQ();
            cb_reserve_back(cb_rstd, onetile);

            add_tiles_init();
            add_tiles(cb_var, cb_eps, first_tile, first_tile, dst0);

            sqrt_tile_init();
            sqrt_tile(dst0);

            pack_tile(dst0, cb_rstd);

            cb_push_back(cb_rstd, onetile);
            REL();
        }
        // We don't pop cb_var here.

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_rstd
         */
        ACQ();
        cb_reserve_back(cb_recip_rstd, onetile);

        add_tiles_init();
        add_tiles(cb_var, cb_eps, first_tile, first_tile, dst0);

        sqrt_tile_init();
        sqrt_tile(dst0);

        recip_tile_init();
        recip_tile(dst0);

        pack_tile(dst0, cb_recip_rstd);

        cb_pop_front(cb_var, onetile);
        cb_push_back(cb_recip_rstd, onetile);
        REL();

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_out
         */
        constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
        cb_wait_front(cb_recip_rstd, onetile);
        for (uint32_t wt = 0; wt < Wt; wt += block_size) {
            cb_reserve_back(cb_gamma_beta_or_out, block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                ACQ();
                if (is_lastdim_layernorm) {
                    mul_bcast_cols_init_short();
                    mul_tiles_bcast_cols(cb_xmm, cb_recip_rstd, wt + j, first_tile, j);
                } else {
                    mul_tiles_bcast_scalar_init_short();
                    mul_tiles_bcast_scalar(cb_xmm, cb_recip_rstd, wt + j, first_tile, j);
                }
                pack_tile(j, cb_gamma_beta_or_out);
                REL();
            }  // block_size loop
            cb_push_back(cb_gamma_beta_or_out, block_size);

            // * gamma
            if (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                cb_wait_front(cb_gamma_beta_or_out, block_size);
                cb_wait_front(cb_gamma, block_size);
                cb_reserve_back(cb_outg, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    ACQ();
                    if (is_groupnorm) {
                        mul_tiles_bcast_scalar_init_short();
                        mul_tiles_bcast_scalar(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            mul_bcast_rows_init_short();
                            mul_tiles_bcast_rows(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        } else {
                            mul_tiles_init();
                            mul_tiles(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        }
                    }
                    pack_tile(j, cb_outg);
                    REL();
                }  // block_size loop
                cb_pop_front(cb_gamma_beta_or_out, block_size);
                cb_pop_front(cb_gamma, block_size);
                cb_push_back(cb_outg, block_size);
            }  // if (gamma_has_value)

            // + beta
            if (beta_has_value) {
                cb_wait_front(cb_gamma_beta, block_size);
                cb_wait_front(cb_beta, block_size);
                cb_reserve_back(cb_out, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    ACQ();
                    if (is_groupnorm) {
                        add_bcast_scalar_init_short();
                        add_tiles_bcast_scalar(cb_gamma_beta, cb_beta, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            add_bcast_rows_init_short();
                            add_tiles_bcast_rows(cb_gamma_beta, cb_beta, j, j, j);
                        } else {
                            add_tiles_init();
                            add_tiles(cb_gamma_beta, cb_beta, j, j, j);
                        }
                    }
                    pack_tile(j, cb_out);
                    REL();
                }  // block_size loop
                cb_pop_front(cb_gamma_beta, block_size);
                cb_pop_front(cb_beta, block_size);
                cb_push_back(cb_out, block_size);
            }  // if (beta_has_value)
        }      // Wt loop
        cb_pop_front(cb_recip_rstd, onetile);
        cb_pop_front(cb_xmm, Wt);
    }  // NCHt loop
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }
}  // void MAIN
}  // namespace NAMESPACE
