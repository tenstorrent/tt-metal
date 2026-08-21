// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) copy_block kernel: process the input in blocks of BLOCK_TILES; per block, copy_block a run
// of consecutive tiles c_0 -> DST, then pack each DST slot -> c_16. Regression baseline for the id-free
// variant copy_block_2_0.cpp (which differs ONLY in the copy_block call). compute_kernel_hw_startup /
// pack_tile are the legacy CB-id API in BOTH kernels so the differential isolates copy_block.
constexpr std::uint32_t BLOCK_TILES = 4;  // fits DST for both fp32 and non-fp32 dest

void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; b += BLOCK_TILES) {
        std::uint32_t ntiles = (per_core_tile_cnt - b) < BLOCK_TILES ? (per_core_tile_cnt - b) : BLOCK_TILES;

        tile_regs_acquire();
        cb0.wait_front(ntiles);
        cb16.reserve_back(ntiles);

        copy_block(tt::CBIndex::c_0, 0, 0, ntiles);

        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t t = 0; t < ntiles; ++t) {
            pack_tile(t, tt::CBIndex::c_16);
        }

        cb0.pop_front(ntiles);
        cb16.push_back(ntiles);
        tile_regs_release();
    }
}
