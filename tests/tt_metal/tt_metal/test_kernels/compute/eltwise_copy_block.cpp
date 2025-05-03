// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);

    for (uint32_t block = 0; block < num_blocks; ++block) {
        acquire_dst();

        // Wait tiles on the input / copy to dst / pop from input
        cb_wait_front(tt::CBIndex::c_0, block_num_tiles);
        for (uint32_t t = 0; t < block_num_tiles; ++t) {
            copy_tile(tt::CBIndex::c_0, t, t);
        }
        cb_pop_front(tt::CBIndex::c_0, block_num_tiles);

        // Reserve space in output / pack / push to output
        cb_reserve_back(tt::CBIndex::c_16, block_num_tiles);
        for (uint32_t t = 0; t < block_num_tiles; ++t) {
            pack_tile(t, tt::CBIndex::c_16);
        }
        cb_push_back(tt::CBIndex::c_16, block_num_tiles);

        release_dst();
    }
}
}  // namespace NAMESPACE
