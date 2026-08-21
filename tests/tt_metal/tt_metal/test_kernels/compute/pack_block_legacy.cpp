// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) pack_block baseline: process the input in blocks of 4 tiles. Per block, copy c_0[0..3] ->
// DST[0..3] with the legacy CB-id copy_tile, then pack the 4-tile block to c_16 with a LOOP of the legacy
// (non-deprecated) CB-id pack_tile(dst_i, c_16) -- in-order packing that auto-advances the packer write pointer
// so tile i lands at consecutive L1 tiles 0..3 in the reserved region. Regression baseline for the id-free
// variant pack_block_2_0.cpp, which differs ONLY in the block pack (experimental::pack_block). copy_tile /
// compute_kernel_hw_startup stay legacy in BOTH kernels so the differential isolates the pack. Compile-time
// arg 0 = total tile count (assumed a multiple of 4). Output must be bit-for-bit identical to the 2.0 kernel.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    constexpr std::uint32_t block = 4;

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; b += block) {
        tile_regs_acquire();
        cb0.wait_front(block);
        cb16.reserve_back(block);

        for (std::uint32_t i = 0; i < block; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }

        tile_regs_commit();
        tile_regs_wait();

        for (std::uint32_t i = 0; i < block; ++i) {
            pack_tile(i, tt::CBIndex::c_16);
        }

        cb0.pop_front(block);
        cb16.push_back(block);
        tile_regs_release();
    }
}
