// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    add_tiles_init(cb_in0, cb_in1);

    // wait for a block of tiles in each of input CBs
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    tile_regs_acquire();  // acquire 8 tile registers

    add_tiles(cb_in0, cb_in1, 0, 0, 0);

    tile_regs_commit();  // signal the packer

    tile_regs_wait();  // packer waits here
    pack_tile(0, cb_out0);
    tile_regs_release();  // packer releases

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    cb_push_back(cb_out0, 1);
}
}  // namespace NAMESPACE
