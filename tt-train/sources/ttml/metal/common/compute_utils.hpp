// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/reconfig_data_format.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/tile_move_copy.h"

inline void pack_and_push(uint32_t reg, uint32_t cb) {
    // NOTE:
    // The order of commit and wait does not matter when adjacent, as they affect different threads.
    // Commit releases the lock for math, letting pack start, while wait ensures math is done.
    // Prefer commit first, then wait, so this function should be called after tile_regs_commit().
    constexpr uint32_t onetile = 1U;
    cb_reserve_back(cb, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, onetile);
}

inline void pack_and_push_block(uint32_t cb_output, uint32_t block_size) {
    // NOTE:
    // Packs multiple tiles (block_size) from consecutive registers to output circular buffer.
    // Should be called after tile_regs_commit() for proper synchronization.
    cb_reserve_back(cb_output, block_size);
    tile_regs_wait();
    pack_reconfig_data_format(cb_output);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output);
    }
    tile_regs_release();
    cb_push_back(cb_output, block_size);
};

inline void zero_dst_reg(const uint32_t i) {
    constexpr float zero = 0.0f;
    fill_tile_init();
    fill_tile(i, zero);
}

inline void pack_and_push_two_blocks(uint32_t cb_output_1, uint32_t cb_output_2, uint32_t block_size) {
    // NOTE:
    // Packs two blocks from consecutive registers to two output circular buffers.
    // Should be called after tile_regs_commit() for proper synchronization.
    cb_reserve_back(cb_output_1, block_size);
    cb_reserve_back(cb_output_2, block_size);
    tile_regs_wait();
    pack_reconfig_data_format(cb_output_1);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output_1);
    }
    pack_reconfig_data_format(cb_output_2);
    for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
        pack_tile(block_idx, cb_output_2);
    }
    tile_regs_release();
    cb_push_back(cb_output_1, block_size);
    cb_push_back(cb_output_2, block_size);
}

// Packs tiles from source circular buffer to output circular buffer.
// Handles data format reconfiguration for both reading and packing.
// Useful for transferring intermediate results to output buffers or between compute stages.
inline void pack_tiles_to_output(const uint32_t cb_source, const uint32_t cb_output, const uint32_t num_tiles) {
    cb_wait_front(cb_source, num_tiles);
    cb_reserve_back(cb_output, num_tiles);

    pack_reconfig_data_format(cb_output);
    reconfig_data_format(cb_source, cb_source);

    copy_tile_init(cb_source);
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        tile_regs_acquire();
        copy_tile(cb_source, tile_idx, /* register idx */ 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(/* register idx */ 0, cb_output);
        tile_regs_release();
    }
    cb_push_back(cb_output, num_tiles);
    cb_pop_front(cb_source, num_tiles);
}
