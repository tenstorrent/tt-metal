// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    copy_tile_init(tt::CBIndex::c_0);
    pack_rows_init(num_rows);

    cb_wait_front(tt::CBIndex::c_0, 1);
    cb_reserve_back(tt::CBIndex::c_16, 1);

    tile_regs_acquire();
    copy_tile(tt::CBIndex::c_0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_rows(0, tt::CBIndex::c_16, 0);
    tile_regs_release();

    pack_rows_uninit();

    cb_pop_front(tt::CBIndex::c_0, 1);
    cb_push_back(tt::CBIndex::c_16, 1);
}
}  // namespace NAMESPACE
