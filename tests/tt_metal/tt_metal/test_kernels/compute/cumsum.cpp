// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/compute/transpose_dest.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/cumsum.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr int onetile = 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t NC = get_compile_time_arg_val(2);

#ifndef ROWWISE
    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    transpose_init(tt::CBIndex::c_0);
#endif
    cumsum_tile_init();

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    for (uint32_t nc = 0; nc < NC; ++nc) {
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            for (uint32_t ht = 0; ht < Ht; ++ht) {
                cb16.reserve_back(onetile);
                tile_regs_acquire();
                cb0.wait_front(onetile);

#ifndef ROWWISE
                copy_tile(tt::CBIndex::c_0, 0, 0);
#else
                transpose_init(tt::CBIndex::c_0);
                transpose_tile(tt::CBIndex::c_0, 0, 0);
#endif
                cumsum_tile(0, ht == 0);
#ifdef ROWWISE
                transpose_dest_init(tt::CBIndex::c_0);
                transpose_dest(0);
#endif

                tile_regs_commit();
                tile_regs_wait();

                pack_tile(0, tt::CBIndex::c_16);

                cb0.pop_front(onetile);
                tile_regs_release();
                cb16.push_back(onetile);
            }
        }
    }
}
