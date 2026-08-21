// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/experimental/2_0/transpose_dest.h"
#include "api/dataflow/circular_buffer.h"

// Id-free (2.0) transpose_dest kernel: per tile, copy c_0 -> DST, in-DST 32x32 transpose, pack -> c_16.
// IDENTICAL to transpose_dest_legacy.cpp except the transpose uses experimental::transpose_dest[_init] (no
// operand arg). copy_tile / pack_tile stay the legacy CB-id API so the differential isolates transpose_dest.
// Output must be bit-for-bit identical to the legacy kernel.
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

        experimental::transpose_dest_init();
        experimental::transpose_dest(0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
