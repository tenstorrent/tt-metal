// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// pool_blocks state copy (data movement only): the raw bytes of a DRAM-interleaved tensor ("state", its pages in
// page-id order from page 0 of the run) to or from consecutive 1024-byte rows of the paged K/V pool [1, 1, NB * bs,
// 512] (ROW_MAJOR bf16).  Pool row r of the run = state bytes [1024 r, min(1024 r + 1024, total)).  A state page of P
// bytes is read or written whole when P <= 1024 (1024 a multiple of P), and in 1024-byte slices of the page when P is a
// multiple of 1024.  A unit is 32 pool rows; core j of n takes the units j, j + n, ...
// Default direction state -> pool; STATE_FROM_POOL reverses it.
// Compile-time args: TensorAccessorArgs state, pool.
// Runtime args: 0 state, 1 pool addresses, 2 total bytes, 3 state page of the run's first byte, 4 state page bytes,
// 5 pool row of the run's first row, 6 core index, 7 core count.  CB 0: one 64-byte aligned 32-row buffer.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t state_addr = get_arg_val<uint32_t>(0);
    const uint32_t pool_addr = get_arg_val<uint32_t>(1);
    const uint32_t total = get_arg_val<uint32_t>(2);
    const uint32_t page0 = get_arg_val<uint32_t>(3);
    const uint32_t page_bytes = get_arg_val<uint32_t>(4);
    const uint32_t row0 = get_arg_val<uint32_t>(5);
    const uint32_t core = get_arg_val<uint32_t>(6);
    const uint32_t cores = get_arg_val<uint32_t>(7);
    constexpr uint32_t ROW_BYTES = 1024, GROUP_ROWS = 32, GROUP_BYTES = GROUP_ROWS * ROW_BYTES;
    constexpr auto state_args = TensorAccessorArgs<0>();
    constexpr auto pool_args = TensorAccessorArgs<state_args.next_compile_time_args_offset()>();
    const auto state = TensorAccessor(state_args, state_addr);
    const auto pool = TensorAccessor(pool_args, pool_addr);

    const uint32_t groups = (total + GROUP_BYTES - 1) / GROUP_BYTES;
    if (core >= groups) {
        return;
    }
    cb_reserve_back(0, 1);
    const uint32_t buffer = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t seg_bytes = page_bytes < ROW_BYTES ? page_bytes : ROW_BYTES;
    for (uint32_t g = core; g < groups; g += cores) {
        const uint32_t begin = g * GROUP_BYTES;
        const uint32_t end = begin + GROUP_BYTES < total ? begin + GROUP_BYTES : total;
#ifdef STATE_FROM_POOL
        for (uint32_t pos = begin; pos < end; pos += ROW_BYTES) {
            const uint32_t bytes = end - pos < ROW_BYTES ? end - pos : ROW_BYTES;
            noc_async_read(pool.get_noc_addr(row0 + (pos >> 10), 0), buffer + (pos - begin), bytes);
        }
        noc_async_read_barrier();
        for (uint32_t pos = begin; pos < end; pos += seg_bytes) {
            noc_async_write(
                buffer + (pos - begin), state.get_noc_addr(page0 + pos / page_bytes, pos % page_bytes), seg_bytes);
        }
        noc_async_write_barrier();
#else
        for (uint32_t pos = begin; pos < end; pos += seg_bytes) {
            noc_async_read(
                state.get_noc_addr(page0 + pos / page_bytes, pos % page_bytes), buffer + (pos - begin), seg_bytes);
        }
        noc_async_read_barrier();
        for (uint32_t pos = begin; pos < end; pos += ROW_BYTES) {
            const uint32_t bytes = end - pos < ROW_BYTES ? end - pos : ROW_BYTES;
            noc_async_write(buffer + (pos - begin), pool.get_noc_addr(row0 + (pos >> 10), 0), bytes);
        }
        noc_async_write_barrier();
#endif
    }
}
