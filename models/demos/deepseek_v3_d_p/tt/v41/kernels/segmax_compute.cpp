// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Segment max of a bf16 row-major row (DeepSeek-V4.1 candidate selection, tt/v41/indexer.py segment_max).
//
// A chunk of BLOCK * 1024 contiguous row elements, tilized as BLOCK tiles of a [32, 32 * BLOCK] row-major block,
// puts superblock (32 consecutive elements) BLOCK * r + j of the chunk in row r of tile j. Per tile:
//   SEG 32: MAX row reduce -> column 0 row r = max of superblock BLOCK * r + j.
//   SEG 8:  for b < 4, tile + mask b (-inf outside columns [8b, 8b + 8), 0 inside) then MAX row reduce ->
//           column 0 row r = max of block 4 (BLOCK * r + j) + b.
// The writer (segmax_writer.cpp) picks column 0 of every result tile. All values stay bf16: x + 0 and x + -inf are
// exact, and the max of bf16 values (scaler 1.0) is exact.
//
// compile_time_args = [SEG, BLOCK]; runtime args = [num_chunks].

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reduce.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t SEG = get_compile_time_arg_val(0);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(1);
    constexpr uint32_t PER_TILE = SEG == 8 ? 4 : 1;  // result tiles per input tile
    const uint32_t num_chunks = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_rm = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;
    constexpr uint32_t cb_mask = tt::CBIndex::c_2;
    constexpr uint32_t cb_tiled = tt::CBIndex::c_3;
    constexpr uint32_t cb_masked = tt::CBIndex::c_4;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_reduce_in = SEG == 8 ? cb_masked : cb_tiled;

    DataflowBuffer rm(cb_rm);
    DataflowBuffer scaler(cb_scaler);
    DataflowBuffer mask(cb_mask);
    DataflowBuffer tiled(cb_tiled);
    DataflowBuffer masked(cb_masked);
    DataflowBuffer out(cb_out);

    compute_kernel_hw_startup(cb_rm, cb_scaler, cb_out);
    scaler.wait_front(1);
    if constexpr (SEG == 8) {
        mask.wait_front(4);
    }

    for (uint32_t c = 0; c < num_chunks; ++c) {
        tilize_init(cb_rm, BLOCK, cb_tiled);
        rm.wait_front(BLOCK);
        tiled.reserve_back(BLOCK);
        tilize_block(cb_rm, BLOCK, cb_tiled);
        tiled.push_back(BLOCK);
        rm.pop_front(BLOCK);
        tilize_uninit(cb_rm, cb_tiled);

        tiled.wait_front(BLOCK);
        if constexpr (SEG == 8) {
            add_tiles_init(cb_tiled, cb_mask);
            masked.reserve_back(BLOCK * 4);
            for (uint32_t j = 0; j < BLOCK; ++j) {
                tile_regs_acquire();
                for (uint32_t b = 0; b < 4; ++b) {
                    add_tiles(cb_tiled, cb_mask, j, b, b);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t b = 0; b < 4; ++b) {
                    pack_tile(b, cb_masked);
                }
                tile_regs_release();
            }
            masked.push_back(BLOCK * 4);
            tiled.pop_front(BLOCK);
            masked.wait_front(BLOCK * 4);
        }

        constexpr uint32_t results = BLOCK * PER_TILE;
        constexpr uint32_t DST = 4;
        out.reserve_back(results);
        reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_reduce_in, cb_scaler, cb_out);
        for (uint32_t t0 = 0; t0 < results; t0 += DST) {
            tile_regs_acquire();
            for (uint32_t i = 0; i < DST; ++i) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(cb_reduce_in, cb_scaler, t0 + i, 0, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < DST; ++i) {
                pack_tile(i, cb_out);
            }
            tile_regs_release();
        }
        reduce_uninit();
        out.push_back(results);
        if constexpr (SEG == 8) {
            masked.pop_front(BLOCK * 4);
        } else {
            tiled.pop_front(BLOCK);
        }
    }
}
