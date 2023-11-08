// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/tile_move_copy.h"

// #include "debug_print.h"

void ACQ() { acquire_dst(tt::DstMode::Half); }
void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    uint32_t step = get_arg_val<uint32_t>(0);
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CB::c_in0;
    constexpr auto cb_grad_in = tt::CB::c_in1;
    constexpr auto cb_exp_avg_in = tt::CB::c_in2;
    constexpr auto cb_exp_avg_sq_in = tt::CB::c_in3;

#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CB::c_in4;
#endif

    constexpr auto cb_param_out = tt::CB::c_out0;
    constexpr auto cb_exp_avg_out = tt::CB::c_out1;
    constexpr auto cb_exp_avg_sq_out = tt::CB::c_out2;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CB::c_out3;
#endif

    constexpr uint32_t cb_lr = tt::CB::c_intermed0;
    constexpr uint32_t cb_beta1 = tt::CB::c_intermed1;
    constexpr uint32_t cb_beta2 = tt::CB::c_intermed2;
    constexpr uint32_t cb_eps = tt::CB::c_intermed3;
    constexpr uint32_t cb_weight_decay = tt::CB::c_intermed4;
    constexpr uint32_t cb_one = tt::CB::c_intermed5;

    constexpr uint32_t cb_tmp1 = tt::CB::c_intermed6;
    constexpr uint32_t cb_tmp2 = tt::CB::c_intermed7;

    constexpr uint32_t dst0 = 0;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_lr, onetile);
    cb_wait_front(cb_beta1, onetile);
    cb_wait_front(cb_beta2, onetile);
    cb_wait_front(cb_eps, onetile);
    cb_wait_front(cb_weight_decay, onetile);
    cb_wait_front(cb_one, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        // grad += grad + param * weight_decay;
        // cb_tmp1 : param * weight_decay;
        ACQ();
        cb_wait_front(cb_param_in, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        mul_tiles_init();
        mul_tiles(cb_param_in, cb_weight_decay, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_grad_in : cb_grad_in + cb_tmp1;
        // ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_reserve_back(cb_grad_in, onetile);
        add_tiles_init();
        add_tiles(cb_grad_in, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_grad_in);
        cb_pop_front(cb_tmp1, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_push_back(cb_grad_in, onetile);
        REL();

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // cb_tmp1 = (1 - beta1)
        ACQ();
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init();
        sub_tiles(cb_one, cb_beta1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_tmp1 = grad * cb_tmp1
        ACQ();
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        mul_tiles_init();
        mul_tiles(cb_grad_in, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_exp_avg_in = cb_exp_avg_in * beta1
        ACQ();
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_reserve_back(cb_exp_avg_in, onetile);
        mul_tiles_init();
        mul_tiles(cb_exp_avg_in, cb_beta1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_in);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_push_back(cb_exp_avg_in, onetile);
        REL();

        // cb_exp_avg_in = cb_exp_avg_in + cb_tmp1
        ACQ();
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_exp_avg_in, onetile);
        add_tiles_init();
        add_tiles(cb_exp_avg_in, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_in);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_push_back(cb_exp_avg_in, onetile);
        cb_pop_front(cb_tmp1, onetile);

        ACQ();
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_reserve_back(cb_exp_avg_out, onetile);
        copy_tile_init();
        copy_tile(cb_exp_avg_in, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_out);
        cb_push_back(cb_exp_avg_out, onetile);
        REL();
        ////////////////////////////////////////////////////////////////////////


        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // cb_tmp1 = (1 - beta2)
        ACQ();
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init();
        sub_tiles(cb_one, cb_beta2, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_tmp2 = grad * grad
        ACQ();
        cb_wait_front(cb_grad_in, onetile);
        cb_reserve_back(cb_tmp2, onetile);
        mul_tiles_init();
        mul_tiles(cb_grad_in, cb_grad_in, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp2);
        cb_push_back(cb_tmp2, onetile);
        REL();

        // cb_tmp1 = cb_tmp1 * cb_tmp2
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        mul_tiles_init();
        mul_tiles(cb_tmp1, cb_tmp2, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_exp_avg_sq_in = cb_exp_avg_sq_in * beta2
        ACQ();
        cb_wait_front(cb_exp_avg_sq_in, onetile);
        cb_reserve_back(cb_exp_avg_sq_in, onetile);
        mul_tiles_init();
        mul_tiles(cb_exp_avg_sq_in, cb_beta2, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_sq_in);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
        cb_push_back(cb_exp_avg_sq_in, onetile);
        REL();

        // cb_exp_avg_sq_in = cb_exp_avg_sq_in + cb_tmp1
        ACQ();
        cb_wait_front(cb_exp_avg_sq_in, onetile);
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_exp_avg_sq_in, onetile);
        add_tiles_init();
        add_tiles(cb_exp_avg_sq_in, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_sq_in);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
        cb_push_back(cb_exp_avg_sq_in, onetile);
        cb_pop_front(cb_tmp1, onetile);
        REL();

        ACQ();
        cb_wait_front(cb_exp_avg_sq_in, onetile);
        cb_reserve_back(cb_exp_avg_sq_out, onetile);
        copy_tile_init();
        copy_tile(cb_exp_avg_sq_in, first_tile, dst0);
        pack_tile(dst0, cb_exp_avg_sq_out);
        cb_push_back(cb_exp_avg_sq_out, onetile);
        REL();
        ////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        // cb_tmp1 = pow(beta2, step);
        ACQ();
        copy_tile_init();
        copy_tile(cb_beta2, first_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        pack_tile(dst0, cb_tmp1);

        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_tmp1 = 1 - cb_tmp1;
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        sub_tiles_init();
        sub_tiles(cb_one, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_tmp1 = 1 / cb_tmp1;
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        copy_tile_init();
        copy_tile(cb_tmp1, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

#ifdef AMSGRAD
        // TODO
        // max_exp_avg_sq = max(max_exp_avg_sq_in, exp_avg_sq);
#endif

        // cb_tmp1 = sqrt(exp_avg_sq / cb_tmp1);
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        mul_tiles_init();
#ifdef AMSGRAD
        mul_tiles(cb_max_exp_avg_sq_in, cb_tmp1, first_tile, first_tile, dst0);
#else
        mul_tiles(cb_exp_avg_sq_in, cb_tmp1, first_tile, first_tile, dst0);
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // cb_tmp1 = 1 / (cb_tmp1 + eps)
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        add_tiles_init();
        add_tiles(cb_tmp1, cb_eps, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // bias_correction1 = 1 - pow(beta1, step);
        // cb_tmp2 = pow(beta1, step);
        ACQ();
        cb_reserve_back(cb_tmp2, onetile);
        copy_tile_init();
        copy_tile(cb_beta1, first_tile, dst0);
        power_tile_init();
        power_tile(dst0, step);
        pack_tile(dst0, cb_tmp2);
        cb_push_back(cb_tmp2, onetile);
        REL();

        // cb_tmp2 = 1 / (1 - cb_tmp2);
        ACQ();
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp2, onetile);
        sub_tiles_init();
        sub_tiles(cb_one, cb_tmp2, first_tile, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_tmp2);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp2, onetile);
        REL();

        // cb_tmp2 = lr * cb_tmp2;
        ACQ();
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp2, onetile);
        mul_tiles_init();
        mul_tiles(cb_lr, cb_tmp2, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp2);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp2, onetile);
        REL();

        // cb_tmp2 = cb_tmp2 * exp_avg;
        ACQ();
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp2, onetile);
        mul_tiles_init();
        mul_tiles(cb_tmp2, cb_exp_avg_in, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp2);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp2, onetile);
        REL();

        // cb_tmp1 = cb_tmp1 * cb_tmp2;
        ACQ();
        cb_wait_front(cb_tmp1, onetile);
        cb_wait_front(cb_tmp2, onetile);
        cb_reserve_back(cb_tmp1, onetile);
        mul_tiles_init();
        mul_tiles(cb_tmp1, cb_tmp2, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_pop_front(cb_tmp1, onetile);
        cb_pop_front(cb_tmp2, onetile);
        cb_push_back(cb_tmp1, onetile);
        REL();

        // param = param - cb_tmp1;
        ACQ();
        cb_reserve_back(cb_param_out, onetile);
        cb_wait_front(cb_tmp1, onetile);
        sub_tiles_init();
        sub_tiles(cb_param_in, cb_tmp1, first_tile, first_tile, dst0);
        pack_tile(dst0, cb_tmp1);
        cb_push_back(cb_param_out, onetile);
        cb_pop_front(cb_tmp1, onetile);
        REL();

        cb_pop_front(cb_param_in, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
    }

    cb_pop_front(cb_lr, onetile);
    cb_pop_front(cb_beta1, onetile);
    cb_pop_front(cb_beta2, onetile);
    cb_pop_front(cb_eps, onetile);
    cb_pop_front(cb_weight_decay, onetile);
    cb_pop_front(cb_one, onetile);
}  // void MAIN
}  // namespace NAMESPACE
