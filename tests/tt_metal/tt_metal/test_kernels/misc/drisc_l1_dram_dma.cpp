// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests multiple DRISC L1 (128KB) to DRAM GDDR Xfer over DMA and measures bandwidth

#include "experimental/core_local_mem.h"
#include "experimental/gddr_dma.h"
#include "risc_common.h"

void writes_cycles_consumed(uint32_t l1_addr, uint32_t offset, uint64_t total_time) {
    experimental::CoreLocalMem<uint64_t> total_time_res(l1_addr);
    total_time_res[offset] = total_time;
}

void kernel_main() {
    // set max burst size
    experimental::dma_set_burst_size(255);

#ifdef L1_TO_GDDR_WRITE_TEST
    uint32_t dst_gddr_addr = get_arg_val<uint32_t>(0);
    const uint32_t l1_src_addr = get_arg_val<uint32_t>(1);
    const uint32_t bytes_per_iter = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);

    constexpr uint32_t tx_stream = 0;
    uint64_t start = get_timestamp();
    // Multiple L1 to DRAM GDDR transfers
    for (uint32_t i = 0; i < iters; i++) {
        experimental::dma_async_write(tx_stream, l1_src_addr, dst_gddr_addr, bytes_per_iter);
        experimental::dma_async_write_barrier(tx_stream);
        dst_gddr_addr += bytes_per_iter;
    }
    uint64_t total_time = get_timestamp() - start;
    // Timing stored immediately after data buffer. Host reads at l1_addr + bytes_per_iter
    uint32_t offset = bytes_per_iter / sizeof(uint64_t);
    writes_cycles_consumed(l1_src_addr, offset, total_time);
#else
    const uint32_t src_gddr_addr = get_arg_val<uint32_t>(0);
    const uint32_t l1_dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t bytes_per_iter = get_arg_val<uint32_t>(2);
    const uint32_t iters = get_arg_val<uint32_t>(3);

    constexpr uint32_t tx_stream = 0;
    uint64_t start = get_timestamp();
    // Same src and dst each iteration - data is overwritten in DRISC L1 in place; pure BW measurement.
    for (uint32_t i = 0; i < iters; i++) {
        experimental::dma_async_read(tx_stream, src_gddr_addr, l1_dst_addr, bytes_per_iter);
        experimental::dma_async_read_barrier(tx_stream);
    }
    uint64_t total_time = get_timestamp() - start;
    // Timing stored immediately after data buffer. Host reads at l1_addr + bytes_per_iter
    uint32_t offset = bytes_per_iter / sizeof(uint64_t);
    writes_cycles_consumed(l1_dst_addr, offset, total_time);
#endif  // L1_TO_GDDR_WRITE_TEST

    experimental::dma_restore_default_transfer_attrs();
}
