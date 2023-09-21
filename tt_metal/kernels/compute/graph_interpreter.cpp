// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"


#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/gelu.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t num_ops = get_compile_time_arg_val(1);

    // Need to pre-initialize an op_info struct and pass into get_next_op_info and modify in that func, since hlkc doesn't support funcs returning vals yet
    tt::op_info_t op_info = {0, 0, 0, 0, 0, 0, 0};
    graph_interpreter_init();

    for (uint32_t op_idx = 0; op_idx < num_ops; op_idx++) {
        get_next_op_info(op_info);

        for (uint32_t idx = 0; idx < per_core_tile_cnt; idx++) {
            cb_reserve_back(op_info.cb_out_id, 1);
            acquire_dst(tt::DstMode::Half);
            cb_wait_front(op_info.cb_in0_id, 1);


            if (op_info.unary) {
                copy_tile_init();
                copy_tile(op_info.cb_in0_id, 0, 0);
            } else {
                cb_wait_front(op_info.cb_in1_id, 1);
            }

            if (op_info.op_code == (int)tt::OpCode::Exponential) { // 0
                exp_tile_init();
                exp_tile(0);
            } else if (op_info.op_code == (int)tt::OpCode::Reciprocal) { // 1
                recip_tile_init();
                recip_tile(0);
            } else if (op_info.op_code == (int)tt::OpCode::Gelu) { // 2
                gelu_tile_init();
                gelu_tile(0, false);
            } else if (op_info.op_code == (int)tt::OpCode::Add) { // 3
                add_tiles_init();
                add_tiles(op_info.cb_in0_id, op_info.cb_in1_id, 0, 0, 0);
            } else if (op_info.op_code == (int)tt::OpCode::Subtract) { // 4
                sub_tiles_init();
                sub_tiles(op_info.cb_in0_id, op_info.cb_in1_id, 0, 0, 0);
            } else if (op_info.op_code == (int)tt::OpCode::Multiply) { // 5
                mul_tiles_init();
                mul_tiles(op_info.cb_in0_id, op_info.cb_in1_id, 0, 0, 0);
            }

            pack_tile(0, op_info.cb_out_id);

            if (op_info.pop0) {
                cb_pop_front(op_info.cb_in0_id, 1);  // Don't always pop, may need the input for later
            }

            if (not op_info.unary and op_info.pop1) {
                cb_pop_front(op_info.cb_in1_id, 1);
            }

            release_dst(tt::DstMode::Half);
            cb_push_back(op_info.cb_out_id, 1);
        }
    }
}
}
