// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr bool divisor_has_value = get_compile_time_arg_val(1) == 1;

    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_divisor = tt::CB::c_in3;
    constexpr uint32_t cb_output_grad = tt::CB::c_in4;
    constexpr uint32_t cb_one = tt::CB::c_in5;
    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_tmp1 = tt::CB::c_intermed1;
    constexpr uint32_t cb_tmp2 = tt::CB::c_intermed2;
    constexpr uint32_t cb_input_grad = tt::CB::c_out0;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_tmp_weight, cb_output_grad);

    cb_wait_front(cb_one, onetile);

    if (divisor_has_value) {
        cb_wait_front(cb_divisor, 1);
        cb_reserve_back(cb_tmp1, onetile);

        ACQ();
        copy_tile_init();
        copy_tile(cb_divisor, first_tile, dst0);
        recip_tile_init();
        recip_tile(dst0);
        pack_tile(dst0, cb_tmp1);
        REL();

        cb_push_back(cb_tmp1, onetile);
    }

    cb_wait_front(cb_output_grad, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_tmp_weight, onetile);

        cb_reserve_back(cb_tmp2, onetile);
        ACQ();
        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast_scalar(cb_tmp_weight, cb_output_grad, first_tile, first_tile, dst0);
        negative_tile_init();
        negative_tile(dst0);
        pack_tile(dst0, cb_tmp2);
        REL();
        cb_push_back(cb_tmp2, onetile);

        if (divisor_has_value) {
            cb_wait_front(cb_tmp1, onetile);
            cb_wait_front(cb_tmp2, onetile);
            cb_reserve_back(cb_input_grad, onetile);
            ACQ();
            mul_tiles_bcast_scalar_init_short();
            mul_tiles_bcast_scalar(cb_tmp2, cb_tmp1, first_tile, first_tile, dst0);
            pack_tile(dst0, cb_input_grad);
            REL();
            cb_push_back(cb_input_grad, onetile);
            cb_pop_front(cb_tmp2, onetile);
        } else {
            cb_wait_front(cb_tmp2, onetile);
            cb_reserve_back(cb_input_grad, onetile);
            ACQ();
            copy_tile_init();
            copy_tile(cb_tmp2, first_tile, dst0);
            pack_tile(dst0, cb_input_grad);
            REL();
            cb_push_back(cb_input_grad, onetile);
            cb_pop_front(cb_tmp2, onetile);
        }

        cb_pop_front(cb_tmp_weight, onetile);
    }

    cb_pop_front(cb_one, onetile);
    if (divisor_has_value) {
        cb_pop_front(cb_divisor, onetile);
    }
}
}  // namespace NAMESPACE
