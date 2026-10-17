// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Pack-side reconfig onto a block-float output CB whose tile is not the full 32x32 tile. c_0 carries the
// payload (Float16_b, same tile shape as c_16) and is copied c_0 -> DST -> c_16 (Bfp8_b); the host decodes
// c_16, so the packer's exponent-section geometry must match the CB's tile.
// The packer starts configured for c_16 itself and pack_reconfig_data_format(c_16) runs before the first
// pack: the unconditional reconfig must keep the geometry hw_configure programmed for the same CB.
void kernel_main() {
    constexpr std::uint32_t onetile = 1;
    const std::uint32_t num_tiles = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);
    pack_reconfig_data_format(tt::CBIndex::c_16);

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        tile_regs_acquire();
        cb0.wait_front(onetile);
        copy_tile(tt::CBIndex::c_0, /*in_tile_index=*/0, /*dst_tile_index=*/0);
        cb16.reserve_back(onetile);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*ifrom_dst=*/0, tt::CBIndex::c_16);

        cb0.pop_front(onetile);
        cb16.push_back(onetile);
        tile_regs_release();
    }
}
