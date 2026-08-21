// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/bcast.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) unary_bcast kernel: per tile, unary_bcast c_0 -> DST (BroadcastType::ROW), pack -> c_16.
// Regression baseline for the id-free variant unary_bcast_2_0.cpp (which differs ONLY in the unary_bcast
// init/op call). hw_startup / pack_tile are the legacy CB-id API in BOTH kernels so the differential isolates
// unary_bcast. ROW broadcast exercises the SrcB / B2D datacopy path (not a plain pass-through copy).
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    unary_bcast_init<BroadcastType::ROW>(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        unary_bcast<BroadcastType::ROW>(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
