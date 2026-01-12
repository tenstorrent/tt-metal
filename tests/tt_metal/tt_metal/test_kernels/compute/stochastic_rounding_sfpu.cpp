// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/rounding.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t seed = get_compile_time_arg_val(1);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
    init_prng_seed(seed);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb16.reserve_back(1);
        acquire_dst();

        cb0.wait_front(1);

        copy_tile(tt::CBIndex::c_0, 0, 0);

        stochastic_round_tile(0);

        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);

        release_dst();
        cb16.push_back(1);
    }
}
}  // namespace NAMESPACE
