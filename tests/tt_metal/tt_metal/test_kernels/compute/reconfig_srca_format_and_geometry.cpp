// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

// SrcA starts out configured for c_1, then the conditional reconfig_data_format_srca(c_1, c_0) switches it to
// c_0, which differs from c_1 in both data format and tile geometry. c_0 is then copied c_0 -> DST -> c_16, so
// the output must equal the input.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    compute_kernel_hw_startup(tt::CBIndex::c_1, tt::CBIndex::c_16);
    reconfig_data_format_srca(tt::CBIndex::c_1, tt::CBIndex::c_0);
    copy_init(tt::CBIndex::c_0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);
    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        copy_tile(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
