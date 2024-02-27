// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/reduce.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_one = tt::CB::c_in3;
    constexpr uint32_t cb_tmp_weight = tt::CB::c_intermed0;
    constexpr uint32_t cb_output = tt::CB::c_out0;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t onetile = 1;

    reduce_init<true>(REDUCE_OP, REDUCE_DIM, cb_tmp_weight, cb_one);
    cb_wait_front(cb_one, onetile);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_tmp_weight, onetile);
        cb_reserve_back(cb_output, onetile);

        ACQ();
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        reduce_tile(REDUCE_OP, REDUCE_DIM, cb_tmp_weight, cb_one, 0, 0, dst0);
        pack_tile(dst0, cb_output);
        reduce_revert_delta();
        REL();

        cb_push_back(cb_output, onetile);
        cb_pop_front(cb_tmp_weight, onetile);
    }

    cb_pop_front(cb_one, onetile);
}
}  // namespace NAMESPACE
