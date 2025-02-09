// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

// #include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    uint32_t third_dim = get_compile_time_arg_val(2);

    tilize_init(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < per_core_block_cnt * per_core_block_tile_cnt * third_dim; ++b) {
        cb_wait_front(tt::CBIndex::c_0, 1);
        cb_reserve_back(tt::CBIndex::c_16, 1);

        tilize_block(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, 1);
        cb_pop_front(tt::CBIndex::c_0, 1);
    }
}
}  // namespace NAMESPACE
