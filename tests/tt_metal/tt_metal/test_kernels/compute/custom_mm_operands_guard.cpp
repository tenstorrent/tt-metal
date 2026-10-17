// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/compressed_custom_mm.h"
#include "api/compute/experimental/custom_mm.h"

// Runs only the CB-id full init of custom_mm (mode 0) or compressed_custom_mm (mode 1) and exits:
// the in0 tile-height asserts in that init are what the guard test observes through the watcher.
// Nothing here waits on another thread, so a tripped assert parks only the thread that hit it and
// the core can be re-initialized afterwards; a thread parked in a tile-register semaphore wait
// cannot be, which is why the product run is not used for the asserting case.
void kernel_main() {
    constexpr std::uint32_t mode = get_compile_time_arg_val(0);  // plain, compressed
    if constexpr (mode == 1) {
        compressed_custom_mm_block_init<false, false, true>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    } else {
        custom_mm_block_init<false, false, true>(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, 1);
    }
}
