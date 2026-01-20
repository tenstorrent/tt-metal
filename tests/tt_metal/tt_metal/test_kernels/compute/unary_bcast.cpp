// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

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
}
}  // namespace NAMESPACE
