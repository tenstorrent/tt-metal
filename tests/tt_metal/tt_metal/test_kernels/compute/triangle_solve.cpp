// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for the per-tile triangle-solve LLK test. Loads RHS into DST[0], runs the SFPU
// forward-substitution solve (triangle_solve_tile) reading the resident unit-lower-tri L
// tile straight from L1, and packs the solution DST[1] into cb_x.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/triangle_solve.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

void kernel_main() {
    constexpr uint32_t cb_l = tt::CBIndex::c_0;    // unit-lower-tri L (resident in L1)
    constexpr uint32_t cb_rhs = tt::CBIndex::c_1;  // RHS
    constexpr uint32_t cb_x = tt::CBIndex::c_2;    // solution X

    compute_kernel_hw_startup(cb_rhs, cb_l, cb_x);

    CircularBuffer cb_l_o(cb_l);
    CircularBuffer cb_rhs_o(cb_rhs);
    CircularBuffer cb_x_o(cb_x);

    cb_l_o.wait_front(1);
    cb_rhs_o.wait_front(1);
    cb_x_o.reserve_back(1);

    tile_regs_acquire();
    copy_tile_init(cb_rhs);
    copy_tile(cb_rhs, 0, /*dst=*/0);  // RHS -> DST[0]
    triangle_solve_tile_init();
    {
        // Profiler zone: measures the solve alone (excludes the RHS copy_tile and the pack).
        DeviceZoneScopedN("TRIANGLE_SOLVE");
        triangle_solve_tile(cb_l_o, /*l_tile_idx=*/0, /*idst_in=*/0, /*idst_out=*/1);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(1, cb_x, 0);  // DST[1] -> cb_x
    tile_regs_release();

    cb_x_o.push_back(1);
    cb_l_o.pop_front(1);
    cb_rhs_o.pop_front(1);
}
