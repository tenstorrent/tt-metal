// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) reconfig kernel: per tile, reconfig SrcA to c_0's OWN format (matched-format reconfig --
// old == new, exercising the "always programs" 1-arg overload without an actual format change), copy
// c_0 -> DST, pack -> c_16. Regression baseline for the id-free variant reconfig_srca_2_0.cpp, which
// differs ONLY in the reconfig_data_format_srca call. copy_tile / pack_tile are the legacy CB-id API in
// BOTH kernels so the differential isolates reconfig_data_format_srca.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    copy_tile_init(tt::CBIndex::c_0);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();
        cb0.wait_front(1);
        cb16.reserve_back(1);

        reconfig_data_format_srca(tt::CBIndex::c_0);
        copy_tile(tt::CBIndex::c_0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
        tile_regs_release();
    }
}
