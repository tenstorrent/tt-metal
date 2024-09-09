// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/cumsum.h"

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

    init_sfpu(tt::CB::c_in0);
    cumsum_tile_init();

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for(uint32_t wt = 0; wt < Wt; ++wt) {
            for(uint32_t ht = 0; ht < Ht; ++ht) {
                cb_reserve_back(tt::CB::c_out0, onetile);
                acquire_dst(tt::DstMode::Half);
                cb_wait_front(tt::CB::c_in0, onetile);

                copy_tile(tt::CB::c_in0, 0, 0);
                cumsum_tile(0, ht == 0);

                pack_tile(0, tt::CB::c_out0);

                cb_pop_front(tt::CB::c_in0, onetile);
                release_dst(tt::DstMode::Half);
                cb_push_back(tt::CB::c_out0, onetile);
            }
        }
    }
}
}
