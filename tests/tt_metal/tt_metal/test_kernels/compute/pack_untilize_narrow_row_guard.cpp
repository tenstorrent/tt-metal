// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"

// Runs only the narrow-row pack_untilize init for ROW_NUM_DATUMS datums per output row and exits: the
// row-byte-width assert in that init is what the guard test observes through the watcher. Nothing here
// waits on another thread, so a tripped assert parks only the pack thread and the core can be
// re-initialized afterwards; a product kernel whose pack init asserts leaves the unpack and math threads
// in a tile-register wait, which blocks the core's re-initialization.
void kernel_main() {
    constexpr std::uint32_t row_num_datums = get_compile_time_arg_val(0);
    constexpr auto in = tt::CBIndex::c_0;
    constexpr auto out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(in, out);
    pack_untilize_dest_init<1 /*block_ct_dim*/, 1 /*full_ct_dim*/, true /*narrow_row*/, row_num_datums>(out);
    pack_untilize_uninit(out);
}
