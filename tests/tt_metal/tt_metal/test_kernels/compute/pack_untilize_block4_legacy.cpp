// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/pack_untilize.h"
#include "api/dataflow/circular_buffer.h"

// Legacy (CB-id) pack-untilize with block_ct_dim = 4 (> 1). Regression baseline for the id-free variant
// pack_untilize_block4_2_0.cpp: both must produce bit-identical row-major output. The legacy path advances
// per-tile input reads by the CB's actual fifo_page_size (one Bfp8 tile == 68 words), so it is the golden
// against which the id-free tile_stride_words is checked.
void kernel_main() {
    constexpr std::uint32_t BLOCK = 4;

    CircularBuffer cb0(tt::CBIndex::c_0);
    CircularBuffer cb16(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    pack_untilize_init<BLOCK, BLOCK>(tt::CBIndex::c_0, tt::CBIndex::c_16);

    cb0.wait_front(BLOCK);
    cb16.reserve_back(BLOCK);
    pack_untilize_block<BLOCK, BLOCK>(tt::CBIndex::c_0, 1 /*block_rt_dim*/, tt::CBIndex::c_16);
    cb0.pop_front(BLOCK);
    cb16.push_back(BLOCK);

    pack_untilize_uninit(tt::CBIndex::c_16);
}
