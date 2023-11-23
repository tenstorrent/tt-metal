// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    const auto num_tiles = get_arg_val<uint32_t>(0);
    const auto p = get_arg_val<uint32_t>(1);
    const bool p_is_negative = get_arg_val<uint32_t>(2) == 1;
    const auto norm_type = get_arg_val<uint32_t>(3);

    constexpr auto cb_input = tt::CB::c_in0;    // input(==tmp_pow_sum)
    constexpr auto cb_decimal = tt::CB::c_in1;  // decimal

    // x^p * exp(log(x) * decimal)
    constexpr auto cb_y = tt::CB::c_out0;  // output(==total_norm)

    constexpr auto cb_x = tt::CB::c_intermed0;         // Sum[tmp_pow_sum](==x)
    constexpr auto cb_xpow = tt::CB::c_intermed1;      // x^p
    constexpr auto cb_logx = tt::CB::c_intermed2;      // log(x)
    constexpr auto cb_exp_lxmd = tt::CB::c_intermed3;  // exp(log(x) * decimal)

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    if (num_tiles > 1) {
        binary_op_init_common(cb_input, cb_x);
    } else {
        binary_op_init_common(cb_logx, cb_decimal);
    }

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    // Compute cb_x
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        if (tile_idx == 0) {
            ACQ();
            cb_wait_front(cb_input, onetile);  // comes from the reader
            cb_reserve_back(cb_x, onetile);

            copy_tile_init();
            copy_tile(cb_input, 0, dst0);

            pack_tile(dst0, cb_x);

            cb_pop_front(cb_input, onetile);
            cb_push_back(cb_x, onetile);
            REL();
        } else {
            ACQ();
            cb_wait_front(cb_input, onetile);  // comes from the reader
            cb_wait_front(cb_x, onetile);
            cb_reserve_back(cb_x, onetile);

            add_tiles_init();
            add_tiles(cb_input, cb_x, 0, 0, dst0);

            pack_tile(dst0, cb_x);

            cb_pop_front(cb_input, onetile);
            cb_pop_front(cb_x, onetile);
            cb_push_back(cb_x, onetile);
            REL();
        }
    }

    // Compute cb_xpow
    // x^p
    ACQ();
    cb_wait_front(cb_x, onetile);
    cb_reserve_back(cb_xpow, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }

    pack_tile(dst0, cb_xpow);

    cb_push_back(cb_xpow, onetile);
    REL();
    // We don't pop cb_x here.

    // Compute cb_logx
    // log(x)
    ACQ();
    cb_reserve_back(cb_logx, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);

    pack_tile(dst0, cb_logx);

    cb_pop_front(cb_x, onetile);
    cb_push_back(cb_logx, onetile);
    REL();

    // Compute cb_exp_lxmd
    // exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_logx, onetile);
    cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init();
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);

    pack_tile(dst0, cb_exp_lxmd);

    cb_pop_front(cb_logx, onetile);
    cb_pop_front(cb_decimal, onetile);
    cb_push_back(cb_exp_lxmd, onetile);
    REL();

    // Compute cb_y
    // x^p * exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_xpow, onetile);
    cb_wait_front(cb_exp_lxmd, onetile);
    cb_reserve_back(cb_y, onetile);

    mul_tiles_init();
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    pack_tile(dst0, cb_y);

    cb_pop_front(cb_xpow, onetile);
    cb_pop_front(cb_exp_lxmd, onetile);
    cb_push_back(cb_y, onetile);
    REL();
}  // void MAIN
}  // namespace NAMESPACE
