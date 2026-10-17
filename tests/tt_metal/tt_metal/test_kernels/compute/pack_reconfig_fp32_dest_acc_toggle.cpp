// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Packer data-format reconfig inside a mid-kernel dest-acc override.
//
// The kernel switches Dest to the mode opposite to the one it was built with
// (set_fp32_dest_acc<!DST_ACCUM_MODE>()), copies each tile into Dest with that mode, reconfigures the
// packer from the Float16_b output configured at startup (never written) to the Bfp8_b output, and
// packs. The reconfig must keep the packer's Dest read width at what the override programmed:
// re-deriving it from the kernel-wide flag makes the packer read 32-bit rows out of a 16-bit Dest
// (fp32 build) or 16-bit halves of 32-bit rows (16-bit build).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr auto cb_in = tt::CBIndex::c_0;        // Float16_b
    constexpr auto cb_out = tt::CBIndex::c_16;      // Bfp8_b, the output that is written
    constexpr auto cb_startup = tt::CBIndex::c_17;  // Float16_b, only configures the packer at startup

    CircularBuffer in(cb_in);
    CircularBuffer out(cb_out);

    // The Dest mode of the rest of the kernel: the opposite of the one it was built with.
    constexpr bool section_fp32 = !DST_ACCUM_MODE;

    compute_kernel_hw_startup(cb_in, cb_startup);
    // Explicit-mode APIs take the overridden mode.
    copy_init<section_fp32>(cb_in);

    set_fp32_dest_acc<section_fp32>(cb_startup);

    for (uint32_t t = 0; t < num_tiles; ++t) {
        in.wait_front(1);
        out.reserve_back(1);

        tile_regs_acquire();
        copy_tile<section_fp32>(cb_in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        // The output formats differ, so this reprograms the packer, Dest read width included.
        pack_reconfig_data_format(cb_startup, cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();

        in.pop_front(1);
        out.push_back(1);
    }

    restore_fp32_dest_acc<section_fp32>(cb_out);
}
