// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) tilize kernel, classic circular buffers. One tile per iteration (block = 1). This is
// the regression baseline for the id-free variant tilize_2_0.cpp: both must produce bit-identical output.
void kernel_main() {
    std::uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

    for (std::uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb0.wait_front(1);
        cb16.reserve_back(1);

        tilize_block(tt::CBIndex::c_0, 1, tt::CBIndex::c_16);

        cb0.pop_front(1);
        cb16.push_back(1);
    }

    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
}
