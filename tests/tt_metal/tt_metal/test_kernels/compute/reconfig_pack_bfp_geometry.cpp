// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// Pack-side reconfig onto an output CB whose tile is not the full 32x32 tile. c_0 carries the payload
// (Float16_b, same tile shape as c_16) and is copied c_0 -> DST -> c_16 (Bfp8_b or Float16_b); the host
// decodes c_16, so the packer's tile geometry (and, for block float, its exponent section) must match the CB.
//   default:               the packer starts configured for c_16 itself and pack_reconfig_data_format(c_16)
//                          runs before the first pack: the unconditional reconfig must program the
//                          exponent-section geometry of the CB's tile.
//   PACK_RECONFIG_TWO_ARG: the packer starts configured for c_2 (never written; the same format as
//                          c_16 but a different tile geometry) and pack_reconfig_data_format<true>(c_2, c_16)
//                          switches it: the conditional reconfig must act on the geometry change even
//                          though the formats match.
//   PACK_RECONFIG_TWO_ARG_PACK_INIT: as PACK_RECONFIG_TWO_ARG but with the default <false>
//                          pack_reconfig_data_format(c_2, c_16) followed by an explicit pack_init(c_16) in place
//                          of is_tile_dim_reconfig_en.
//   PACK_HW_STARTUP_ONLY:  the packer is configured for c_16 by the hw startup only (control).
void kernel_main() {
    constexpr std::uint32_t onetile = 1;
    const std::uint32_t num_tiles = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);
#if defined(PACK_RECONFIG_TWO_ARG)
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2);
    copy_init(tt::CBIndex::c_0);
    pack_reconfig_data_format<true /*is_tile_dim_reconfig_en*/>(tt::CBIndex::c_2, tt::CBIndex::c_16);
#elif defined(PACK_RECONFIG_TWO_ARG_PACK_INIT)
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2);
    copy_init(tt::CBIndex::c_0);
    pack_reconfig_data_format(tt::CBIndex::c_2, tt::CBIndex::c_16);
    pack_init(tt::CBIndex::c_16);
#elif defined(PACK_HW_STARTUP_ONLY)
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);
#else
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_init(tt::CBIndex::c_0);
    pack_reconfig_data_format(tt::CBIndex::c_16);
#endif

    for (std::uint32_t t = 0; t < num_tiles; ++t) {
        tile_regs_acquire();
        cb0.wait_front(onetile);
        copy_tile(tt::CBIndex::c_0, /*in_tile_index=*/0, /*dst_tile_index=*/0);
        cb16.reserve_back(onetile);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*ifrom_dst=*/0, tt::CBIndex::c_16);

        cb0.pop_front(onetile);
        cb16.push_back(onetile);
        tile_regs_release();
    }
}
