// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    untilize_init(tt::CBIndex::c_0);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb0.wait_front(per_core_block_tile_cnt);
        cb16.reserve_back(per_core_block_tile_cnt);

        untilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        cb16.push_back(per_core_block_tile_cnt);
        cb0.pop_front(per_core_block_tile_cnt);
    }

    untilize_uninit(tt::CBIndex::c_0);
}
}  // namespace NAMESPACE
