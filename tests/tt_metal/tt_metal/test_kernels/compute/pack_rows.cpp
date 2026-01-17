// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

namespace NAMESPACE {

void MAIN {
    constexpr uint32_t num_rows = get_compile_time_arg_val(0);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    copy_tile_init(tt::CBIndex::c_0);
    pack_rows_init(num_rows);

    cb0.wait_front(1);
    cb16.reserve_back(1);

    tile_regs_acquire();
    copy_tile(tt::CBIndex::c_0, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_rows(0, tt::CBIndex::c_16, 0);
    tile_regs_release();

    pack_rows_uninit();

    cb0.pop_front(1);
    cb16.push_back(1);
}
}  // namespace NAMESPACE
