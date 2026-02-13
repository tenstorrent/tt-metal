// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"

/**
 * Pack a block of num_tiles from regs 0..num_tiles-1 into an already-reserved CB. No reserve or push.
 * Call after tile_regs_commit() and tile_regs_wait(); releases at the end.
 */
inline void pack_no_push_block(uint32_t cb_idx, uint32_t num_tiles) {
    pack_reconfig_data_format(cb_idx);
    for (uint32_t i = 0U; i < num_tiles; ++i) {
        pack_tile(i, cb_idx);
    }
    tile_regs_release();
}

/**
 * Pack one tile from reg to cb, reserve+wait+pack+push.
 * NOTE: Call after tile_regs_commit(). The order of commit and wait does not matter when adjacent
 * (they affect different threads). Commit releases the lock for math so pack can start; wait
 * ensures math is done. Prefer commit first, then wait.
 */
inline void pack_and_push(uint32_t reg, uint32_t cb) {
    constexpr uint32_t onetile = 1U;
    cb_reserve_back(cb, onetile);
    tile_regs_wait();
    pack_reconfig_data_format(cb);
    pack_tile(reg, cb);
    tile_regs_release();
    cb_push_back(cb, onetile);
}

/**
 * Pack a block of n tiles from consecutive DEST registers (0..block_size-1), reserve+wait+pack+push.
 * NOTE: Call after tile_regs_commit(). The order of commit and wait does not matter when adjacent
 * (they affect different threads). Commit releases the lock for math so pack can start; wait
 * ensures math is done. Prefer commit first, then wait.
 */
inline void pack_and_push_block(uint32_t cb_output, uint32_t block_size) {
    cb_reserve_back(cb_output, block_size);
    tile_regs_wait();
    pack_no_push_block(cb_output, block_size);
    cb_push_back(cb_output, block_size);
}

/**
 * Pack a block of tiles from consecutive DEST registers to a CB using L1 accumulation.
 * Does NOT push to CB; caller must cb_push_back after all blocks are accumulated.
 * Call after tile_regs_commit() and tile_regs_wait(); releases tile regs at the end.
 *
 * @param cb_idx Output circular buffer
 * @param first_block true for the first block (zero accumulator), false to accumulate into existing CB data
 * @param num_tiles Number of tiles to pack (consecutive registers 0..num_tiles-1)
 * @param dst_start_index First output tile index in the reserved CB region
 */
inline void pack_l1_acc_block(uint32_t cb_idx, bool first_block, uint32_t num_tiles, uint32_t dst_start_index) {
    pack_reconfig_data_format(cb_idx);
    pack_reconfig_l1_acc(first_block ? 0U : 1U);
    for (uint32_t k = 0U; k < num_tiles; ++k) {
        pack_tile<true>(k, cb_idx, dst_start_index + k);
    }
    pack_reconfig_l1_acc(0U);
    tile_regs_release();
}

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
