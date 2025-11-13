// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_rows.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint_tensix.h"
#include "debug/dprint_pages.h"
namespace NAMESPACE {

void MAIN {
    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows_to_pack = get_compile_time_arg_val(2);
    constexpr uint32_t row_num_datums = get_compile_time_arg_val(3);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    copy_tile_to_dst_init_short(tt::CBIndex::c_0);

    pack_rows_init<row_num_datums>(tt::CBIndex::c_16, num_rows_to_pack);

    for (uint32_t r = 0; r < per_core_block_cnt; ++r) {
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            cb_wait_front(tt::CBIndex::c_0, 1);

            acquire_dst();

            copy_tile(tt::CBIndex::c_0, 0, 0);

            pack_rows<row_num_datums>(0, tt::CBIndex::c_16);

            release_dst();

            cb_pop_front(tt::CBIndex::c_0, 1);
        }

        PACK((tt::compute::common::print_tile_rows(tt::CBIndex::c_16, 16, 0, 0)));
        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
    }
}

}  // namespace NAMESPACE
