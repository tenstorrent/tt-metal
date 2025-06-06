// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <compute_kernel_api/cb_api.h>
#include <compute_kernel_api/pack.h>
#include <compute_kernel_api/reconfig_data_format.h>
#include <compute_kernel_api/reg_api.h>
#include <debug/dprint.h>
#include <hostdevcommon/kernel_structs.h>
#include <tensix.h>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);  // rows to process in this kernel
constexpr uint32_t Wt = get_compile_time_arg_val(1);

constexpr auto cb_first_input = tt::CBIndex::c_0;
constexpr auto cb_second_input = tt::CBIndex::c_1;
constexpr auto cb_output = tt::CBIndex::c_2;

constexpr uint32_t onetile = 1U;

void MAIN {
    init_sfpu(cb_first_input, cb_output);
    binary_op_init_common(cb_first_input, cb_second_input, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        cb_wait_front(cb_first_input, Wt);   //  wait Wt tiles
        cb_wait_front(cb_second_input, Wt);  //  wait Wt tiles

        const uint32_t first_register = 0;
        const uint32_t second_register = 1U;

        cb_reserve_back(cb_output, Wt);
        reconfig_data_format(cb_first_input, cb_second_input);
        for (uint32_t col = 0; col < Wt; col++) {
            tile_regs_acquire();
            copy_tile_init(cb_first_input);
            copy_tile(cb_first_input, /* tile_idx */ col, /* register_idx */ first_register);

            copy_tile_init(cb_second_input);
            copy_tile(cb_second_input, /* tile_idx */ col, /* register_idx */ second_register);

            max_tile_init();
            max_tile(first_register, second_register);
            tile_regs_commit();

            tile_regs_wait();
            pack_reconfig_data_format(cb_output);
            pack_tile(first_register, cb_output);
            tile_regs_release();
            cb_push_back(cb_output, onetile);
        }

        cb_pop_front(cb_first_input, Wt);
        cb_pop_front(cb_second_input, Wt);
    }
}

}  // namespace NAMESPACE
