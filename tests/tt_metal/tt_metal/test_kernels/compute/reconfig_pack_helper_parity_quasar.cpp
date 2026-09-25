// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

void kernel_main() {
    constexpr std::uint32_t input_cb = tt::CBIndex::c_0;
    constexpr std::uint32_t output_cb = tt::CBIndex::c_16;

    compute_kernel_hw_startup(input_cb, output_cb);
    copy_init(input_cb);
    for (std::uint32_t section = 0; section < 2; ++section) {
        cb_wait_front(input_cb, 1);
        tile_regs_acquire();
        copy_tile(input_cb, 0, 0);
        tile_regs_commit();
        pack_and_push(0, output_cb);
        cb_pop_front(input_cb, 1);
    }
}
