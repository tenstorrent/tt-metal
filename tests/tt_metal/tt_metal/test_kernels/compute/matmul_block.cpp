// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/compute_kernel_hw_startup.h"

namespace NAMESPACE {
void MAIN {
    uint32_t block_tile_dim = get_compile_time_arg_val(0);
    uint32_t dst_tile_rows = get_compile_time_arg_val(1);
    uint32_t dst_tile_cols = get_compile_time_arg_val(2);
    uint32_t block_cnt = get_compile_time_arg_val(3);
    uint32_t in0_block_tile_cnt = get_compile_time_arg_val(4);
    uint32_t in1_block_tile_cnt = get_compile_time_arg_val(5);
    uint32_t out_block_tile_cnt = get_compile_time_arg_val(6);

#if (TEST_INIT_SHORT == 1)
#if (WITH_DT == 1)
    // Intentionally wrong init with different data formats
    matmul_block_init(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_16,
        false,
        dst_tile_cols - 1,
        dst_tile_rows - 1,
        block_tile_dim - 1);
    // Corrected init short with dt
    matmul_block_init_reconfig_data_format(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#elif (WITH_DT == 0)
    // Intentionally wrong init with same data formats
    matmul_block_init(
        tt::CBIndex::c_1,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        false,
        dst_tile_cols - 1,
        dst_tile_rows - 1,
        block_tile_dim - 1);
    // Corrected init short
    matmul_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif
#elif (TEST_INIT_SHORT == 0)
    // Hardware startup - common MMIO configurations
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

    // Initialize matmul block operation
    matmul_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif

    acquire_dst();
    for (uint32_t b = 0; b < block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, in0_block_tile_cnt);
        cb_wait_front(tt::CBIndex::c_1, in1_block_tile_cnt);

        matmul_block(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false, dst_tile_cols, dst_tile_rows, block_tile_dim);

        cb_pop_front(tt::CBIndex::c_0, in0_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_1, in1_block_tile_cnt);
    }

    // Pack out
    cb_reserve_back(tt::CBIndex::c_16, out_block_tile_cnt);
    for (uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, tt::CBIndex::c_16);
    }
    cb_push_back(tt::CBIndex::c_16, out_block_tile_cnt);

    release_dst();
}
}  // namespace NAMESPACE
