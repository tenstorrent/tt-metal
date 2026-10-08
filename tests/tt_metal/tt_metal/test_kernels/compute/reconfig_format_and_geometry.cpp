// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// One unpacker starts out configured for c_2, then a conditional reconfig switches it to a CB that differs from
// c_2 in both data format and tile geometry, and c_0 is passed through to c_16 unchanged:
//   default:        reconfig_data_format_srca(c_2, c_0), then copy c_0 -> DST -> c_16.
//   RECONFIG_SRCB:  SrcA is set up for c_0 and SrcB for c_2; reconfig_data_format_srcb(c_2, c_1), then
//                   c_0 + c_1 -> c_16, where the host fills c_1 with zeros.
void kernel_main() {
    constexpr std::uint32_t onetile = 1;
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);
#ifdef RECONFIG_SRCB
    CircularBuffer cb1(tt::CBIndex::c_1);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_16);
    reconfig_data_format_srcb(tt::CBIndex::c_2, tt::CBIndex::c_1);
    add_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
#else
    compute_kernel_hw_startup(tt::CBIndex::c_2, tt::CBIndex::c_16);
    reconfig_data_format_srca(tt::CBIndex::c_2, tt::CBIndex::c_0);
    copy_init(tt::CBIndex::c_0);
#endif

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(onetile);
#ifdef RECONFIG_SRCB
        cb1.wait_front(onetile);
        add_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, /*itile0=*/0, /*itile1=*/0, /*idst=*/0);
#else
        copy_tile(tt::CBIndex::c_0, /*in_tile_index=*/0, /*dst_tile_index=*/0);
#endif
        cb16.reserve_back(onetile);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/*dst_tile_index=*/0, tt::CBIndex::c_16);

        cb0.pop_front(onetile);
#ifdef RECONFIG_SRCB
        cb1.pop_front(onetile);
#endif
        cb16.push_back(onetile);
        tile_regs_release();
    }
}
