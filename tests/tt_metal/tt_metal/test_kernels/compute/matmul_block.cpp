// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#endif

void kernel_main() {
    uint32_t block_tile_dim = get_compile_time_arg_val(0);
    uint32_t dst_tile_rows = get_compile_time_arg_val(1);
    uint32_t dst_tile_cols = get_compile_time_arg_val(2);
    uint32_t block_cnt = get_compile_time_arg_val(3);
    uint32_t in0_block_tile_cnt = get_compile_time_arg_val(4);
    uint32_t in1_block_tile_cnt = get_compile_time_arg_val(5);
    uint32_t out_block_tile_cnt = get_compile_time_arg_val(6);

#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb0(0);
    experimental::DataflowBuffer dfb1(1);
    experimental::DataflowBuffer dfb_out(2);
#endif

#if (TEST_INIT_SHORT == 1)
#if (WITH_DT == 1)
    // Intentionally wrong init with different data formats
#ifdef ARCH_QUASAR
    // The TEST_INIT_SHORT WITH_DT path is not implemented for Quasar yet
#else
    mm_block_init(
        tt::CBIndex::c_0,
        tt::CBIndex::c_2,
        tt::CBIndex::c_16,
        false,
        dst_tile_cols - 1,
        dst_tile_rows - 1,
        block_tile_dim - 1);
    // Corrected init short with dt
    mm_block_init_short_with_dt(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_2, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif
#elif (WITH_DT == 0)
    // Intentionally wrong init with same data formats
#ifdef ARCH_QUASAR
    mm_block_init(
        dfb1.get_id(),
        dfb0.get_id(),
        dfb_out.get_id(),
        false,
        dst_tile_cols - 1,
        dst_tile_rows - 1,
        block_tile_dim - 1);
    mm_block_init_short(dfb0.get_id(), dfb1.get_id(), false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#else
    mm_block_init(
        tt::CBIndex::c_1,
        tt::CBIndex::c_0,
        tt::CBIndex::c_16,
        false,
        dst_tile_cols - 1,
        dst_tile_rows - 1,
        block_tile_dim - 1);
    // Corrected init short
    mm_block_init_short(tt::CBIndex::c_0, tt::CBIndex::c_1, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif
#endif
#elif (TEST_INIT_SHORT == 0)
#ifdef ARCH_QUASAR
    mm_block_init(dfb0.get_id(), dfb1.get_id(), dfb_out.get_id(), false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#else
    mm_block_init(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, false, dst_tile_cols, dst_tile_rows, block_tile_dim);
#endif
#endif

    acquire_dst();
    for (uint32_t b = 0; b < block_cnt; ++b) {
#ifdef ARCH_QUASAR
        dfb0.wait_front(in0_block_tile_cnt);
        dfb1.wait_front(in1_block_tile_cnt);

        matmul_block(dfb0.get_id(), dfb1.get_id(), 0, 0, 0, false, dst_tile_cols, dst_tile_rows, block_tile_dim);

        dfb0.pop_front(in0_block_tile_cnt);
        dfb1.pop_front(in1_block_tile_cnt);
#else
        cb_wait_front(tt::CBIndex::c_0, in0_block_tile_cnt);
        cb_wait_front(tt::CBIndex::c_1, in1_block_tile_cnt);

        matmul_block(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, false, dst_tile_cols, dst_tile_rows, block_tile_dim);

        cb_pop_front(tt::CBIndex::c_0, in0_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_1, in1_block_tile_cnt);
#endif
    }

    // Pack out
#ifdef ARCH_QUASAR
    dfb_out.reserve_back(out_block_tile_cnt);
    for (uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, dfb_out.get_id());
    }
    dfb_out.push_back(out_block_tile_cnt);
#else
    cb_reserve_back(tt::CBIndex::c_16, out_block_tile_cnt);
    for (uint32_t i = 0; i < out_block_tile_cnt; ++i) {
        pack_tile(i, tt::CBIndex::c_16);
    }
    cb_push_back(tt::CBIndex::c_16, out_block_tile_cnt);
#endif
    release_dst();
}
