// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-only probe for max_reduce_with_indices. The host test
// (tests/tt_metal/tt_metal/llk/test_max_reduce_with_indices_guard.cpp) compiles this kernel
// with MPWI_NUM_ROWS / MPWI_ACCUMULATE defines and asserts whether the build succeeds, so
// the template guards in the SFPU kernel are exercised without running anything on the device.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reg_api.h"

void kernel_main() {
    max_reduce_with_indices_init<ckernel::DataLayout::ROW_MAJOR>();
    tile_regs_acquire();
    max_reduce_with_indices<MPWI_NUM_ROWS, ckernel::DataLayout::ROW_MAJOR, MPWI_ACCUMULATE>(
        /*idst=*/0, /*idst_idx=*/2, /*chunk=*/0);
    tile_regs_commit();
    tile_regs_wait();
    tile_regs_release();
}
