// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DRISC Test Kernel
// Tests DRISC L1 (128KB) to DRAM GDDR Xfer over DMA

#include "api/compile_time_args.h"
#include "experimental/core_local_mem.h"
#include "experimental/gddr_dma.h"
#include "risc_common.h"

void kernel_main() {
    // set max burst size
    experimental::dma_set_burst_size(255);

#ifdef L1_TO_GDDR_WRITE_TEST
    uint32_t dst_gddr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_src_addr = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_iter = get_compile_time_arg_val(2);
    constexpr uint32_t iters = get_compile_time_arg_val(3);

    constexpr uint32_t tx_stream = 0;
    uint64_t start = get_timestamp();
    for (uint32_t i = 0; i < iters; i++) {
        experimental::dma_async_write(tx_stream, l1_src_addr, dst_gddr_addr, bytes_per_iter);
        experimental::dma_async_read_barrier(tx_stream);
        dst_gddr_addr += bytes_per_iter;
    }
    uint64_t end = get_timestamp();
    uint64_t total_time = end - start;
    experimental::CoreLocalMem<uint64_t> total_time_res(l1_src_addr);
    uint32_t offset = (bytes_per_iter) / sizeof(uint64_t);
    total_time_res[offset] = total_time;
#else
    constexpr uint32_t src_gddr_addr = get_compile_time_arg_val(0);
    constexpr uint32_t l1_dst_addr = get_compile_time_arg_val(1);
    constexpr uint32_t bytes_per_iter = get_compile_time_arg_val(2);
    constexpr uint32_t iters = get_compile_time_arg_val(3);

    constexpr uint32_t tx_stream = 0;
    uint64_t start = get_timestamp();
    // Repeatedly write to the same L1 addr
    for (uint32_t i = 0; i < iters; i++) {
        experimental::dma_async_read(tx_stream, src_gddr_addr, l1_dst_addr, bytes_per_iter);
        experimental::dma_async_read_barrier(tx_stream);
    }
    uint64_t end = get_timestamp();
    uint64_t total_time = end - start;
    experimental::CoreLocalMem<uint64_t> total_time_res(l1_dst_addr);
    uint32_t offset = (bytes_per_iter) / sizeof(uint64_t);
    total_time_res[offset] = total_time;
#endif  // L1_TO_GDDR_WRITE_TEST

    experimental::dma_restore_default_transfer_attrs();
}
