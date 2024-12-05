// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
#ifndef SHORT_INIT
    untilize_init(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    untilize_init_short(tt::CBIndex::c_0);
#endif

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        untilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }

    untilize_uninit(tt::CBIndex::c_0);
}
}  // namespace NAMESPACE
