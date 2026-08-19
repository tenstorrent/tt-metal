// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) eltwise-binary ADD kernel, classic circular buffers, one tile per iteration. This is the
// regression baseline for the id-free variant eltwise_binary_add_idfree.cpp: both must produce
// bit-identical output.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb0 = tt::CBIndex::c_0;
    constexpr auto cb1 = tt::CBIndex::c_1;
    constexpr auto cb16 = tt::CBIndex::c_16;

    CircularBuffer c0(cb0);
    CircularBuffer c1(cb1);
    CircularBuffer c16(cb16);

    compute_kernel_hw_startup(cb0, cb1, cb16);
    add_init(cb0, cb1);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        c0.wait_front(1);
        c1.wait_front(1);
        c16.reserve_back(1);

        tile_regs_acquire();
        add_tiles(cb0, cb1, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb16);
        tile_regs_release();

        c0.pop_front(1);
        c1.pop_front(1);
        c16.push_back(1);
    }
}
