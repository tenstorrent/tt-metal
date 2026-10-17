// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Float32 output packed after a mid-kernel dest-acc disable.
//
// Built with fp32_dest_acc_en = true and a Float32 output (a Float32 pack source). The tile is copied into
// the 32-bit Dest, then set_fp32_dest_acc<false>() switches the kernel to a 16-bit Dest before the tile is
// packed. The datums in Dest are still 32-bit, so the packer must keep its 32-bit Dest read for the Float32
// source, as configure_pack programs it, whatever the new mode.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr auto cb_in = tt::CBIndex::c_0;    // Float16_b
    constexpr auto cb_out = tt::CBIndex::c_16;  // Float32

    CircularBuffer in(cb_in);
    CircularBuffer out(cb_out);

    compute_kernel_hw_startup(cb_in, cb_out);
    copy_init(cb_in);

    in.wait_front(1);
    out.reserve_back(1);

    tile_regs_acquire();
    copy_tile(cb_in, 0, 0);  // written while Dest is 32-bit
    tile_regs_commit();

    set_fp32_dest_acc<false>(cb_out);

    tile_regs_wait();
    pack_tile<false, false /*is_fp32_dest_acc_en*/>(0, cb_out);
    tile_regs_release();

    in.pop_front(1);
    out.push_back(1);

    restore_fp32_dest_acc<false>(cb_out);
}
