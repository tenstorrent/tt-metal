// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

    // The following sequence of operations are compiled onto the 3 compute cores (Unpack, Math, Pack) in the Tensix
    // core. The work together to perform the addition of two input tiles and store the result in the output tile to the
    // output circular buffer. Which is then picked up by the writer kernel and written back to DRAM.

    // Metalium API Calls                              Involved Cores
    binary_op_init_common(cb_in0, cb_in1, cb_out0);  // Unpack, Math, Pack
    add_tiles_init(cb_in0, cb_in1);                  // Unpack, Math

    // wait for a tile to be ready in the input CBs
    cb_wait_front(cb_in0, 1);  // Unpack
    cb_wait_front(cb_in1, 1);  // Unpack

    // acquire 8 tile registers to perform the addition
    tile_regs_acquire();  // Math

    // Take data from cb_in0 offset 0th page and
    // cb_in1 offset 0th page. Add them together
    // and store the result in cb_out0 (as
    // configured) offset 0th page.
    add_tiles(cb_in0, cb_in1, 0, 0, 0);  // Unpack, Math

    // signal the packer
    tile_regs_commit();  // Math

    // packer waits here
    tile_regs_wait();  // Pack
    // Copy the result from tile registers to the
    // output circular buffer (also called packing)
    pack_tile(0, cb_out0);  // Pack
    // packer releases
    tile_regs_release();  // Pack

    cb_pop_front(cb_in0, 1);  // Unpack
    cb_pop_front(cb_in1, 1);  // Unpack

    cb_push_back(cb_out0, 1);  // Pack
}
}  // namespace NAMESPACE
