// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_pages.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_inp0 = cb_in0;
    constexpr auto cb_inp1 = cb_in1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr auto cb_in2 = tt::CBIndex::c_2;

    UNPACK((DPRINT << "per_core_block_cnt: " << per_core_block_cnt << ENDL()));
    UNPACK((DPRINT << "per_core_block_size: " << per_core_block_size << ENDL()));

    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);
    sub_bcast_row_tile_init();

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            sub_bcast_row_tile(cb_inp0, cb_inp1, 0, 0, i /*dst_index*/);
            // dprint_tensix_dest_reg<true>(0);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            // dprint_tensix_dest_reg<true>(i);
            for (uint32_t j = 0; j < 10000; j++) {
                TTI_NOP;
            }
            PACK((tt::compute::common::print_full_tile(cb_out0, i)));
        }

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
