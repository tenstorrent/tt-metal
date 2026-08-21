// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) pack-untilize-DEST kernel: per tile, copy c_0 -> DST, then pack_untilize_dest packs
// (untilizes) the tile straight out of the DEST register to c_16. Regression baseline for the id-free variant
// pack_untilize_dest_2_0.cpp (which differs ONLY in the pack_untilize_dest[_init] calls). copy_tile stays the
// legacy CB-id API in BOTH kernels so the differential isolates pack_untilize_dest.
// block_ct_dim = full_ct_dim = block_rt_dim = 1 (one tile per CB slot).
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_to_dst_init_short(tt::CBIndex::c_0);
    pack_untilize_dest_init<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(tt::CBIndex::c_16);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tile_regs_acquire();
        copy_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_untilize_dest<1 /*block_ct_dim*/, 1 /*full_ct_dim*/>(tt::CBIndex::c_16, 1 /*block_rt_dim*/);
        tile_regs_release();

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    pack_untilize_uninit(tt::CBIndex::c_16);
}
