// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reg_api.h"

inline void pack_and_push(uint32_t reg, uint32_t cb) {
    // NOTE:
    // The order of commit and wait does not matter when they are next to each other, as they handle different
    // threads. Commit releases the lock for the math thread, allowing the pack thread to start working on the
    // data, while wait is for the pack thread to finish math. In principle, you can commit first and then wait,
    // or wait first and then commit. Logically, it makes sense to say the math procedure is finished (commit)
    // and then packing can start (wait), so commit first and then wait is preferred. That's why the call of this
    // function should be preceded with tile_regs_commit().
    constexpr uint32_t onetile = 1U;
    cb_reserve_back(cb, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, onetile);
}

inline void pack_and_push_block(uint32_t cb_output, uint32_t block_size) {
    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output);
    }
    tile_regs_release();
    cb_push_back(cb_output, block_size);
};
