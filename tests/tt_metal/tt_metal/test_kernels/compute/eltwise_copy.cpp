// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        acquire_dst();

        // Pop tile after tile, copy to DST and pack
        cb0.wait_front(1);
        cb16.reserve_back(1);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);

        release_dst();
    }
}
}  // namespace NAMESPACE
