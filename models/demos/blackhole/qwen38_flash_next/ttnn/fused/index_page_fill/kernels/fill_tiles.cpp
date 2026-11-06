// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// index_page_fill (data movement only): n tile-rows (4 bf16 tiles of 32 x 32 across the 128 width) of a TILE source
// [1, 1, n * 32, 128] into the paged index pool [NB, 1, block_rows, 128] (TILE).  The start position P (a multiple of
// 128) is read from a device uint32 scalar; source tile-row i is logical tile-row L = (P >> 7) + i, which lands in
// physical tile-row pt[lane][L >> tpb_shift] * tpb + (L & (tpb - 1)), tpb = block_rows / 32 = 1 << tpb_shift.  Physical
// tile id = physical tile-row * 4 + column tile.  Whole tiles only.  Core j of n takes tile-rows j, j + n, ...
// Compile-time args: TensorAccessorArgs src, pool, page table, position.
// Runtime args: 0 src, 1 pool, 2 page table, 3 position addresses, 4 tile-rows, 5 lane, 6 tpb_shift, 7 max_blocks,
// 8 core index, 9 core count.  CB 0: one 64-byte aligned region = the 4-tile buffer, the page table row, the scalar.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pool_addr = get_arg_val<uint32_t>(1);
    const uint32_t pt_addr = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t tile_rows = get_arg_val<uint32_t>(4);
    const uint32_t lane = get_arg_val<uint32_t>(5);
    const uint32_t tpb_shift = get_arg_val<uint32_t>(6);
    const uint32_t max_blocks = get_arg_val<uint32_t>(7);
    const uint32_t core = get_arg_val<uint32_t>(8);
    const uint32_t cores = get_arg_val<uint32_t>(9);
    constexpr uint32_t TILE_BYTES = 2048, ROW_TILES = 4;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto pool_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<pt_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto pool = TensorAccessor(pool_args, pool_addr);
    const auto pt = TensorAccessor(pt_args, pt_addr);
    const auto pos = TensorAccessor(pos_args, pos_addr);

    if (core >= tile_rows) {
        return;
    }
    cb_reserve_back(0, 1);
    const uint32_t buffer = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t pt_l1 = buffer + ROW_TILES * TILE_BYTES;
    const uint32_t pos_l1 = pt_l1 + ((max_blocks * 4 + 63) & ~63u);
    noc_async_read(pt.get_noc_addr(lane, 0), pt_l1, max_blocks * 4);
    noc_async_read_page(0, pos, pos_l1);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* table = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(pt_l1);
    const uint32_t l0 = *reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(pos_l1) >> 7;
    const uint32_t tpb_mask = (1u << tpb_shift) - 1;
    for (uint32_t i = core; i < tile_rows; i += cores) {
        const uint32_t logical = l0 + i;
        const uint32_t dst_row = table[logical >> tpb_shift] * (1u << tpb_shift) + (logical & tpb_mask);
        for (uint32_t c = 0; c < ROW_TILES; ++c) {
            noc_async_read_page(i * ROW_TILES + c, src, buffer + c * TILE_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < ROW_TILES; ++c) {
            noc_async_write_page(dst_row * ROW_TILES + c, pool, buffer + c * TILE_BYTES);
        }
        noc_async_write_barrier();
    }
}
