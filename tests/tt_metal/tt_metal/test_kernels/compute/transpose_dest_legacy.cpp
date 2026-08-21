// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) transpose_dest kernel: per tile, copy c_0 -> DST, in-DST 32x32 transpose, pack -> c_16.
// Regression baseline for the id-free variant transpose_dest_2_0.cpp (which differs ONLY in the transpose_dest
// call). copy_tile / pack_tile are the legacy CB-id API in BOTH kernels so the differential isolates
// transpose_dest. Both re-init copy (datacopy math) and transpose (transpose math) each iteration since the two
// ops alternate on the MATH engine.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        copy_tile_init(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        transpose_dest_init(tt::CBIndex::c_0);
        transpose_dest(0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
