// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the alias-dest-reuse repro.
//
// Pushes:
//   - cb_bcast_a: 1 tile, once
//   - cb_bcast_b: 1 tile, once
//   - cb_primary: block_size tiles per block, n_blocks blocks
//   - cb_alias:   pushed alongside cb_primary. In Case B (single shared L1
//                 allocation) the host-side reservation of cb_alias is on the
//                 same memory, but the alias has its own front/back pointers,
//                 so we still must reserve/push it to satisfy compute's
//                 cb_wait_front(cb_alias, ...). In Case A (separate allocation)
//                 we copy the same x bytes a second time into cb_alias.
//
// Runtime args (in order):
//   0: x_addr           (DRAM addr of x tiles, n_blocks * block_size tiles)
//   1: bcast_a_addr     (DRAM addr of bcast_a, 1 tile)
//   2: bcast_b_addr     (DRAM addr of bcast_b, 1 tile)
//   3: n_blocks
//   4: block_size
//   5: write_alias_separately  (1 = Case A: also async_read into cb_alias's L1;
//                                0 = Case B: cb_alias shares cb_primary's L1)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t x_addr = get_arg_val<uint32_t>(0);
    uint32_t bcast_a_addr = get_arg_val<uint32_t>(1);
    uint32_t bcast_b_addr = get_arg_val<uint32_t>(2);
    uint32_t n_blocks = get_arg_val<uint32_t>(3);
    uint32_t block_size = get_arg_val<uint32_t>(4);
    uint32_t write_alias_separately = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_primary = tt::CBIndex::c_0;
    constexpr uint32_t cb_alias = tt::CBIndex::c_29;
    constexpr uint32_t cb_bcast_a = tt::CBIndex::c_2;
    constexpr uint32_t cb_bcast_b = tt::CBIndex::c_3;

    const uint32_t tile_size_x = get_tile_size(cb_primary);
    const uint32_t tile_size_a = get_tile_size(cb_bcast_a);
    const uint32_t tile_size_b = get_tile_size(cb_bcast_b);

    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto a_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto x_acc = TensorAccessor(x_args, x_addr, tile_size_x);
    const auto a_acc = TensorAccessor(a_args, bcast_a_addr, tile_size_a);
    const auto b_acc = TensorAccessor(b_args, bcast_b_addr, tile_size_b);

    // Push bcast operands once.
    cb_reserve_back(cb_bcast_a, 1);
    uint32_t a_l1 = get_write_ptr(cb_bcast_a);
    noc_async_read_tile(0, a_acc, a_l1);
    noc_async_read_barrier();
    cb_push_back(cb_bcast_a, 1);

    cb_reserve_back(cb_bcast_b, 1);
    uint32_t b_l1 = get_write_ptr(cb_bcast_b);
    noc_async_read_tile(0, b_acc, b_l1);
    noc_async_read_barrier();
    cb_push_back(cb_bcast_b, 1);

    // Stream x tiles block by block.
    uint32_t tile_id = 0;
    for (uint32_t blk = 0; blk < n_blocks; ++blk) {
        cb_reserve_back(cb_primary, block_size);
        if (write_alias_separately) {
            cb_reserve_back(cb_alias, block_size);
        }

        uint32_t primary_l1 = get_write_ptr(cb_primary);
        uint32_t alias_l1 = 0;
        if (write_alias_separately) {
            alias_l1 = get_write_ptr(cb_alias);
        }

        for (uint32_t i = 0; i < block_size; ++i) {
            noc_async_read_tile(tile_id + i, x_acc, primary_l1 + i * tile_size_x);
            if (write_alias_separately) {
                noc_async_read_tile(tile_id + i, x_acc, alias_l1 + i * tile_size_x);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_primary, block_size);
        if (write_alias_separately) {
            cb_push_back(cb_alias, block_size);
        } else {
            // Case B: cb_alias shares cb_primary's L1, but the alias still has
            // its own reservation/front counters. Reserve and push without
            // touching memory (already populated via cb_primary above).
            cb_reserve_back(cb_alias, block_size);
            cb_push_back(cb_alias, block_size);
        }
        tile_id += block_size;
    }
}
