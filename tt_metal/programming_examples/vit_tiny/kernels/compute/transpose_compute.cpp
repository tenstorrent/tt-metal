// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Within-tile WH transpose: transposes each 32x32 tile.
// Reader provides tiles in transposed layout order; this kernel
// transposes the data within each tile.

#include <cstdint>
#include "api/compute/transpose_wh.h"
#include "api/compute/pack.h"

using namespace ckernel;

void kernel_main() {
    const uint32_t n_tiles = get_compile_time_arg_val(0);

    constexpr tt::CBIndex cb_in = tt::CBIndex::c_0;
    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;

    transpose_wh_init(cb_in, cb_out);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_in, 1);
        tile_regs_acquire();
        transpose_wh_tile(cb_in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        cb_pop_front(cb_in, 1);
    }
}
