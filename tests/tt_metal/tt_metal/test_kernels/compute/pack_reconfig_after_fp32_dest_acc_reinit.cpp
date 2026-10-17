// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A full re-init inside a mid-kernel dest-acc override ends it.
//
// Built with fp32_dest_acc_en = true. The kernel disables dest-acc with set_fp32_dest_acc<false>() and,
// without a matching restore, re-initializes the pipeline for the kernel-wide mode: compute_kernel_hw_startup
// (REINIT_WITH_CLEANUP = 0) or compute_kernel_hw_cleanup, Blackhole only (REINIT_WITH_CLEANUP = 1). It then
// copies a tile into the 32-bit Dest, reconfigures the packer from the Float16_b output configured at
// startup to the Bfp8_b output and packs. The reconfig must program the kernel-wide 32-bit Dest read: a
// stale 16-bit override reads 16-bit halves of the 32-bit rows.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#if REINIT_WITH_CLEANUP
#include "api/compute/experimental/compute_kernel_hw_cleanup.h"
#endif

void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;        // Float16_b
    constexpr auto cb_out = tt::CBIndex::c_16;      // Bfp8_b, the output that is written
    constexpr auto cb_startup = tt::CBIndex::c_17;  // Float16_b, only configures the packer at startup

    CircularBuffer in(cb_in);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_in, cb_startup);
    set_fp32_dest_acc<false>(cb_startup);

    // Back to the kernel-wide mode without a restore.
#if REINIT_WITH_CLEANUP
    compute_kernel_hw_cleanup();
#else
    compute_kernel_hw_startup(cb_in, cb_startup);
#endif
    copy_init(cb_in);

    in.wait_front(1);
    out.reserve_back(1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_reconfig_data_format<true>(cb_out);
    pack_tile(0, cb_out);
    tile_regs_release();

    in.pop_front(1);
    out.push_back(1);
}
