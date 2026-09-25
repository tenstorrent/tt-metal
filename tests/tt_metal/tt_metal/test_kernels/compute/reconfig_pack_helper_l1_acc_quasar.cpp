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

    // Prime the one-slot output ring with a value that the first accumulation block must overwrite.
    cb_wait_front(input_cb, 1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    tile_regs_commit();
    pack_and_push(0, output_cb);
    cb_pop_front(input_cb, 1);

    cb_reserve_back(output_cb, 1);
    cb_wait_front(input_cb, 1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    tile_regs_commit();
    pack_l1_acc_block(output_cb, true, 1, 0);
    cb_pop_front(input_cb, 1);

    cb_wait_front(input_cb, 1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    tile_regs_commit();
    pack_l1_acc_block(output_cb, false, 1, 0);
    cb_push_back(output_cb, 1);
    cb_pop_front(input_cb, 1);

    // A normal pack after accumulation proves that the shared helper restored L1_ACC to disabled.
    cb_wait_front(input_cb, 1);
    tile_regs_acquire();
    copy_tile(input_cb, 0, 0);
    tile_regs_commit();
    pack_and_push(0, output_cb);
    cb_pop_front(input_cb, 1);
}
