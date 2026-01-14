// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/transpose_wh_dest.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/cumsum.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

#ifndef ROWWISE
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);
#endif
    cumsum_tile_init();

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                cb16.reserve_back(onetile);
                acquire_dst();
                cb0.wait_front(onetile);

#ifndef ROWWISE
                copy_tile(tt::CBIndex::c_0, 0, 0);
#else
                transpose_wh_init_short(tt::CBIndex::c_0);
                transpose_wh_tile(tt::CBIndex::c_0, 0, 0);
#endif
                cumsum_tile(0, ht == 0);
#ifdef ROWWISE
                transpose_wh_dest_init_short();
                transpose_wh_dest(0);
#endif

                pack_tile(0, tt::CBIndex::c_16);

                cb0.pop_front(onetile);
                release_dst();
                cb16.push_back(onetile);
            }
        }
    }
}
}  // namespace NAMESPACE
