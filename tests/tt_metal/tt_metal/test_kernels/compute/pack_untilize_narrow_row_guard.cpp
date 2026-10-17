// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"

// Runs only the narrow-row pack_untilize init for row_num_datums datums per output row across full_ct_dim
// tile columns in blocks of block_ct_dim tiles, and exits: the narrow-row init's LLK_ASSERTs (row byte width and
// block geometry on Blackhole, face geometry on Wormhole) are what the assert-fixture tests observe through the
// watcher, and the compile-contract test compiles it for the widths and block splits the init's static_asserts
// must accept or reject. Nothing here waits on another thread, so a tripped assert
// parks only the pack thread and the core can be re-initialized afterwards; a product kernel whose pack init asserts
// leaves the unpack and math threads in a tile-register wait, which blocks the core's re-initialization.
void kernel_main() {
    constexpr std::uint32_t row_num_datums = get_compile_time_arg_val(0);
    constexpr std::uint32_t full_ct_dim = get_compile_time_arg_val(1);
    constexpr std::uint32_t block_ct_dim = get_compile_time_arg_val(2);
    constexpr auto in = tt::CBIndex::c_0;
    constexpr auto out = tt::CBIndex::c_16;

    compute_kernel_hw_startup(in, out);
    pack_untilize_dest_init<block_ct_dim, full_ct_dim, true /*narrow_row*/, row_num_datums>(out);
    pack_untilize_uninit(out);
}
