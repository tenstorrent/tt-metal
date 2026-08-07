// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sync.hpp"
#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)                        \
    X(uint32_t, progress_counter_addr) \
    X(uint32_t, num_bytes_per_send)    \
    X(uint32_t, transfer_size)         \
    X(uint32_t, send_delta_addr)       \
    X(uint32_t, send_buffer0)          \
    X(uint32_t, send_buffer1)          \
    X(uint32_t, recv_buffer0)          \
    X(uint32_t, recv_buffer1)          \
    X(uint32_t, dram_start_addr)       \
    X(uint32_t, dram_end_addr)         \
    X(uint32_t, dram_bank_id)

void kernel_main() {
    ARG_INIT(ARGS);

    uint32_t* progress_counter = (uint32_t*)progress_counter_addr;
    *progress_counter = 0;

    uint32_t send_buffer = send_buffer0;
    uint32_t send_buffer_next = send_buffer1;

    uint32_t recv_buffer = recv_buffer0;
    uint32_t recv_buffer_next = recv_buffer1;

    uint64_t start = timestamp();

    uint32_t curr_addr = dram_start_addr;

    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, curr_addr);
    noc_async_read(noc_addr, send_buffer, transfer_size);
    noc_async_read_barrier();
    curr_addr += transfer_size;

    for (; curr_addr < dram_end_addr; curr_addr += transfer_size) {
        *progress_counter = curr_addr;
        uint64_t noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, curr_addr);
        noc_async_read(noc_addr, send_buffer_next, transfer_size);

        eth_send_bytes(send_buffer, recv_buffer, transfer_size, num_bytes_per_send, num_bytes_per_send >> 4);

        eth_wait_for_receiver_done();
        noc_async_read_barrier();

        std::swap(send_buffer, send_buffer_next);
        std::swap(recv_buffer, recv_buffer_next);
    }

    eth_send_bytes(send_buffer, recv_buffer, transfer_size, num_bytes_per_send, num_bytes_per_send >> 4);

    eth_wait_for_receiver_done();

    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_addr = delta;
    *progress_counter = dram_end_addr;
}
