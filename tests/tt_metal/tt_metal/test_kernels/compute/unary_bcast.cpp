// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#ifdef ARCH_QUASAR
#include "api/compute/unary_bcast_quasar.h"
#else
#include "api/compute/bcast.h"
#endif
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#ifdef ARCH_QUASAR
    constexpr uint32_t src0_dfb_id = 0;
    constexpr uint32_t src1_dfb_id = 1;
    constexpr uint32_t dst0_dfb_id = 2;
    constexpr uint32_t dst1_dfb_id = 3;

    experimental::DataflowBuffer dfb_src0(src0_dfb_id);
    experimental::DataflowBuffer dfb_dst0(dst0_dfb_id);
    experimental::DataflowBuffer dfb_src1(src1_dfb_id);
    experimental::DataflowBuffer dfb_dst1(dst1_dfb_id);

    unary_bcast_init<BCAST_DIM_0>(src0_dfb_id, dst0_dfb_id);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        dfb_src0.wait_front(per_core_block_dim);
        acquire_dst();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            unary_bcast<BCAST_DIM_0>(src0_dfb_id, tile_index, tile_index);
        }

        dfb_src0.pop_front(per_core_block_dim);
        dfb_dst0.reserve_back(per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            pack_tile(tile_index, dst0_dfb_id);
        }

        dfb_dst0.push_back(per_core_block_dim);
        release_dst();
    }

    reconfigure_unary_bcast<BCAST_DIM_0, BCAST_DIM_1>(src0_dfb_id, src1_dfb_id, dst0_dfb_id, dst1_dfb_id);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        dfb_src1.wait_front(per_core_block_dim);
        acquire_dst();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            unary_bcast<BCAST_DIM_1>(src1_dfb_id, tile_index, tile_index);
        }

        dfb_src1.pop_front(per_core_block_dim);
        dfb_dst1.reserve_back(per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            pack_tile(tile_index, dst1_dfb_id);
        }

        dfb_dst1.push_back(per_core_block_dim);
        release_dst();
    }
#else
    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);
    experimental::CircularBuffer cb1(tt::CBIndex::c_1);
    experimental::CircularBuffer cb17(tt::CBIndex::c_17);

    unary_bcast_init<BCAST_DIM_0>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb0.wait_front(per_core_block_dim);
        acquire_dst();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            unary_bcast<BCAST_DIM_0>(tt::CBIndex::c_0, tile_index, tile_index);
        }

        cb0.pop_front(per_core_block_dim);
        cb16.reserve_back(per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            pack_tile(tile_index, tt::CBIndex::c_16);
        }

        cb16.push_back(per_core_block_dim);
        release_dst();
    }

    reconfigure_unary_bcast<BCAST_DIM_0, BCAST_DIM_1>(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, tt::CBIndex::c_17);

    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb1.wait_front(per_core_block_dim);
        acquire_dst();
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            unary_bcast<BCAST_DIM_1>(tt::CBIndex::c_1, tile_index, tile_index);
        }

        cb1.pop_front(per_core_block_dim);
        cb17.reserve_back(per_core_block_dim);

        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            pack_tile(tile_index, tt::CBIndex::c_17);
        }

        cb17.push_back(per_core_block_dim);
        release_dst();
    }
#endif
}
