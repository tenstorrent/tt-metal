// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    int i{0};
    const auto num_input_tiles_per_core = get_arg_val<uint32_t>(i++);

    const bool need_to_bcast_n = get_arg_val<uint32_t>(i++) == 1;
    const bool need_to_bcast_c = get_arg_val<uint32_t>(i++) == 1;
    const bool need_to_bcast_ht = get_arg_val<uint32_t>(i++) == 1;
    const bool need_to_bcast_wt = get_arg_val<uint32_t>(i++) == 1;

    const auto p = get_arg_val<uint32_t>(i++);
    const bool p_is_negative = get_arg_val<uint32_t>(i++) == 1;
    const auto p_minus_one = get_arg_val<uint32_t>(i++);
    const bool p_minus_one_is_negative = get_arg_val<uint32_t>(i++) == 1;

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;        // input(==x)
    const auto cb_y = input_id++;        // output(==y)
    const auto cb_dy = input_id++;       // output_grad(==dy)
    const auto cb_decimal = input_id++;  // decimal

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_dx = output_id++;  // input_grad(==dx)

    std::uint8_t intermed_id{tt::CB::c_intermed0};
    const auto cb_tmp0 = intermed_id++;
    const auto cb_tmp1 = intermed_id++;
    const auto cb_tmp2 = intermed_id++;
    const auto cb_tmp3 = intermed_id++;
    const auto cb_tmp4 = intermed_id++;
    const auto cb_tmp5 = intermed_id++;
    const auto cb_tmp6 = intermed_id++;

    const auto cb_xpow = cb_tmp0;
    const auto cb_logx = cb_tmp1;
    const auto cb_exp_lxmd = cb_tmp2;
    const auto cb_correct_xpow = cb_tmp3;
    const auto cb_recip_ypow = cb_tmp6;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    cb_wait_front(cb_decimal, onetile);  // comes from the reader

    for (uint32_t idx = 0; idx < num_input_tiles_per_core; ++idx) {
        cb_wait_front(cb_x, onetile);   // comes from the reader
        cb_wait_front(cb_y, onetile);   // comes from the reader
        cb_wait_front(cb_dy, onetile);  // comes from the reader

        // x^(p - 1)
        power_tile_to_cb(
            cb_x, cb_xpow, cb_logx, cb_decimal, cb_exp_lxmd, cb_correct_xpow, p_minus_one, p_minus_one_is_negative);

        // x^(p - 1) * y -> cb_tmp4
        ACQ();
        cb_wait_front(cb_correct_xpow, onetile);
        cb_reserve_back(cb_tmp4, onetile);

        if (need_to_bcast_ht && need_to_bcast_wt) {
            mul_tiles_bcast_scalar_init_short();
            mul_tiles_bcast_scalar(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else if (need_to_bcast_ht) {
            mul_bcast_rows_init_short();
            mul_tiles_bcast_rows(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else if (need_to_bcast_wt) {
            mul_bcast_cols_init_short();
            mul_tiles_bcast_cols(cb_correct_xpow, cb_y, 0, 0, dst0);
        } else {
            mul_tiles_init();
            mul_tiles(cb_correct_xpow, cb_y, 0, 0, dst0);
        }

        pack_tile(dst0, cb_tmp4);

        cb_pop_front(cb_correct_xpow, onetile);
        cb_push_back(cb_tmp4, onetile);
        REL();

        // x^(p - 1) * y * dy -> cb_tmp5
        ACQ();
        cb_wait_front(cb_tmp4, onetile);
        cb_reserve_back(cb_tmp5, onetile);

        if (need_to_bcast_ht && need_to_bcast_wt) {
            mul_tiles_bcast_scalar_init_short();
            mul_tiles_bcast_scalar(cb_tmp4, cb_dy, 0, 0, dst0);
        } else if (need_to_bcast_ht) {
            mul_bcast_rows_init_short();
            mul_tiles_bcast_rows(cb_tmp4, cb_dy, 0, 0, dst0);
        } else if (need_to_bcast_wt) {
            mul_bcast_cols_init_short();
            mul_tiles_bcast_cols(cb_tmp4, cb_dy, 0, 0, dst0);
        } else {
            mul_tiles_init();
            mul_tiles(cb_tmp4, cb_dy, 0, 0, dst0);
        }

        pack_tile(dst0, cb_tmp5);

        cb_pop_front(cb_tmp4, onetile);
        cb_push_back(cb_tmp5, onetile);
        REL();

        // 1 / y^p
        power_and_recip_tile_to_cb(cb_y, cb_tmp0, cb_tmp1, cb_decimal, cb_tmp2, cb_recip_ypow, p, p_is_negative);

        // (x^(p - 1) * y * dy) / y^p -> cb_dx
        ACQ();
        cb_wait_front(cb_tmp5, onetile);
        cb_wait_front(cb_recip_ypow, onetile);
        cb_reserve_back(cb_dx, onetile);

        if (need_to_bcast_ht && need_to_bcast_wt) {
            mul_tiles_bcast_scalar_init_short();
            mul_tiles_bcast_scalar(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else if (need_to_bcast_ht) {
            mul_bcast_rows_init_short();
            mul_tiles_bcast_rows(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else if (need_to_bcast_wt) {
            mul_bcast_cols_init_short();
            mul_tiles_bcast_cols(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        } else {
            mul_tiles_init();
            mul_tiles(cb_tmp5, cb_recip_ypow, 0, 0, dst0);
        }

        pack_tile(dst0, cb_dx);

        cb_pop_front(cb_tmp5, onetile);
        cb_pop_front(cb_recip_ypow, onetile);
        cb_push_back(cb_dx, onetile);
        REL();

        cb_pop_front(cb_dy, onetile);
    }

    cb_pop_front(cb_decimal, onetile);

}  // void MAIN
}  // namespace NAMESPACE
