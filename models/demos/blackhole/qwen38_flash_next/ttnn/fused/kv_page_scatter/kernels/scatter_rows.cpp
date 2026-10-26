// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// kv_page_scatter (data movement only): rows [0, 32 * groups) of a ROW_MAJOR bf16 source cache [1, 1, C, 512] into the
// paged KV pool [1, 1, num_pages * bs, 512] at the physical rows of one page-table row: row t -> pt[lane][t >> shift] *
// bs + (t & (bs - 1)), whole 32-row groups (bs is a power of two >= 32, so a group never crosses a page).  Core j of
// n takes groups j, j + n, ...  The page table row (int32 ROW_MAJOR [1, 1, rows, max_blocks]) is read into L1 once.
// Compile-time args: TensorAccessorArgs src, pool, page table.
// Runtime args: 0 src, 1 pool, 2 page table addresses, 3 groups, 4 lane, 5 shift, 6 bs, 7 max_blocks, 8 core index,
// 9 core count, 10 source row offset (a multiple of 32; 0 = the head of the source).  CB 0: one 64-byte aligned region
// = the 32-row group buffer, then the page table row.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t pool_addr = get_arg_val<uint32_t>(1);
    const uint32_t pt_addr = get_arg_val<uint32_t>(2);
    const uint32_t groups = get_arg_val<uint32_t>(3);
    const uint32_t lane = get_arg_val<uint32_t>(4);
    const uint32_t shift = get_arg_val<uint32_t>(5);
    const uint32_t bs = get_arg_val<uint32_t>(6);
    const uint32_t max_blocks = get_arg_val<uint32_t>(7);
    const uint32_t core = get_arg_val<uint32_t>(8);
    const uint32_t cores = get_arg_val<uint32_t>(9);
    const uint32_t src_row0 = get_arg_val<uint32_t>(10);
    constexpr uint32_t ROW_BYTES = 1024, GROUP_ROWS = 32;
    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr auto pool_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<pool_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(src_args, src_addr);
    const auto pool = TensorAccessor(pool_args, pool_addr);
    const auto pt = TensorAccessor(pt_args, pt_addr);

    if (core >= groups) {
        return;
    }
    cb_reserve_back(0, 1);
    const uint32_t buffer = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t pt_l1 = buffer + GROUP_ROWS * ROW_BYTES;
    noc_async_read(pt.get_noc_addr(lane, 0), pt_l1, max_blocks * 4);
    noc_async_read_barrier();
    invalidate_l1_cache();
    const volatile tt_l1_ptr uint32_t* table = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(pt_l1);
    for (uint32_t g = core; g < groups; g += cores) {
        const uint32_t t0 = g * GROUP_ROWS;
        const uint32_t dst0 = table[t0 >> shift] * bs + (t0 & (bs - 1));
        for (uint32_t k = 0; k < GROUP_ROWS; ++k) {
            noc_async_read(src.get_noc_addr(src_row0 + t0 + k, 0), buffer + k * ROW_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < GROUP_ROWS; ++k) {
            noc_async_write(buffer + k * ROW_BYTES, pool.get_noc_addr(dst0 + k, 0), ROW_BYTES);
        }
        noc_async_write_barrier();
    }
}
