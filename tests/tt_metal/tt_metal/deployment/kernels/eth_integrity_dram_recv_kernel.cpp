// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "sync.hpp"

#define ARGS(X)                        \
    X(uint32_t, transfer_size)         \
    X(uint32_t, recv_buffer0)          \
    X(uint32_t, recv_buffer1)          \
    X(uint32_t, progress_counter_addr) \
    X(uint32_t, dram_start_addr)       \
    X(uint32_t, dram_end_addr)         \
    X(uint32_t, dram_bank_id)

void kernel_main() {
    ARG_INIT(ARGS);

    uint32_t recv_buffer = recv_buffer0;
    uint32_t recv_buffer_next = recv_buffer1;
    uint32_t* progress_counter = (uint32_t*)progress_counter_addr;

    *progress_counter = 0;

    eth_wait_for_bytes(transfer_size);
    eth_receiver_done();

    uint32_t curr_addr = dram_start_addr;
    for (; curr_addr < dram_end_addr - transfer_size; curr_addr += transfer_size) {
        *progress_counter = curr_addr;

        uint64_t noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, curr_addr);
        noc_async_write(recv_buffer, noc_addr, transfer_size);

        eth_wait_for_bytes(transfer_size);

        eth_receiver_done();
        noc_async_write_barrier();

        std::swap(recv_buffer, recv_buffer_next);
    }

    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, curr_addr);
    noc_async_write(recv_buffer, noc_addr, transfer_size);
    noc_async_write_barrier();

    *progress_counter = dram_end_addr;
}
