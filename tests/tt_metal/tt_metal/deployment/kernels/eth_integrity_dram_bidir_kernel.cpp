// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "args.hpp"
#include "timestamp.hpp"

#define ARGS(X)                     \
    X(uint32_t, num_bytes_per_send) \
    X(uint32_t, transfer_size)      \
    X(uint32_t, iter_l1_address)    \
    X(uint32_t, dram_start_addr)    \
    X(uint32_t, dram_end_addr)      \
    X(uint32_t, send_delta_addr)    \
    X(uint32_t, sendbuffer_0)       \
    X(uint32_t, sendbuffer_1)       \
    X(uint32_t, recvbuffer_0)       \
    X(uint32_t, recvbuffer_1)

#define RUNTIME_ARGS(X)    \
    X(uint32_t, channel0)  \
    X(uint32_t, channel1)  \
    X(uint32_t, read_bank) \
    X(uint32_t, write_bank)

void noc_barrier() {
    noc_async_full_barrier();
    noc_async_full_barrier(1);
}

uint32_t dram_to_buf(
    uint32_t read_bank, uint32_t end_addr, uint32_t curraddr, uint32_t transfer_size, uint32_t sendbuf) {
    uint32_t remaining_send = end_addr - curraddr;
    uint32_t to_send = remaining_send > transfer_size ? transfer_size : remaining_send;

    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(read_bank, curraddr);
    noc_async_read(noc_addr, sendbuf, to_send);

    return to_send;
}

uint32_t buf_to_dram(
    uint32_t write_bank, uint32_t end_addr, uint32_t curraddr, uint32_t transfer_size, uint32_t recvbuf) {
    uint32_t remaining_send = end_addr - curraddr;
    uint32_t to_send = remaining_send > transfer_size ? transfer_size : remaining_send;

    uint64_t noc_addr = get_noc_addr_from_bank_id<true>(write_bank, curraddr, 1);
    noc_async_write(recvbuf, noc_addr, to_send);

    return to_send;
}

uint32_t buf_to_eth(
    uint32_t chan,
    uint32_t end_addr,
    uint32_t curraddr,
    uint32_t transfer_size,
    uint32_t sendbuf,
    uint32_t recvbuf,
    uint32_t send_size) {
    /* ============== */
    uint32_t remaining_send = end_addr - curraddr;
    uint32_t to_send = remaining_send > transfer_size ? transfer_size : remaining_send;
    eth_send_bytes_over_channel(sendbuf, recvbuf, to_send, chan, send_size, send_size >> 4);

    return to_send;
}

void kernel_main() {
    ARG_INIT(ARGS);
    ARG_RUNTIME_INIT(RUNTIME_ARGS);

    uint32_t sendbuf0 = sendbuffer_0;
    uint32_t sendbuf1 = sendbuffer_1;
    uint32_t recvbuf0 = recvbuffer_0;
    uint32_t recvbuf1 = recvbuffer_1;

    volatile uint32_t* iter = (volatile uint32_t*)iter_l1_address;

    *iter = 0;

    uint64_t start = timestamp();

    uint32_t curr_send_addr = dram_start_addr;
    uint32_t curr_recv_addr = dram_start_addr;

    curr_send_addr += dram_to_buf(read_bank, dram_end_addr, curr_send_addr, transfer_size, sendbuf1);
    curr_send_addr += dram_to_buf(read_bank, dram_end_addr, curr_send_addr, transfer_size, sendbuf0);
    noc_async_read_barrier();

    uint32_t to_recv =
        buf_to_eth(channel0, dram_end_addr, curr_recv_addr, transfer_size, sendbuf1, recvbuf0, num_bytes_per_send);
    eth_wait_for_bytes_on_channel(to_recv, channel1);
    eth_receiver_channel_done(channel1);
    eth_wait_for_receiver_channel_done(channel0);

    while (curr_send_addr < dram_end_addr) {
        *iter = curr_recv_addr;

        curr_send_addr += dram_to_buf(read_bank, dram_end_addr, curr_send_addr, transfer_size, sendbuf1);

        uint32_t to_recv = buf_to_eth(
            channel0,
            dram_end_addr,
            curr_recv_addr + transfer_size,
            transfer_size,
            sendbuf0,
            recvbuf1,
            num_bytes_per_send);

        curr_recv_addr += buf_to_dram(write_bank, dram_end_addr, curr_recv_addr, transfer_size, recvbuf0);

        noc_barrier();

        eth_wait_for_bytes_on_channel(to_recv, channel1);
        eth_receiver_channel_done(channel1);
        eth_wait_for_receiver_channel_done(channel0);

        std::swap(sendbuf0, sendbuf1);
        std::swap(recvbuf0, recvbuf1);
    }

    to_recv = buf_to_eth(
        channel0, dram_end_addr, curr_recv_addr + transfer_size, transfer_size, sendbuf0, recvbuf1, num_bytes_per_send);

    curr_recv_addr += buf_to_dram(write_bank, dram_end_addr, curr_recv_addr, transfer_size, recvbuf0);

    noc_barrier();

    eth_wait_for_bytes_on_channel(to_recv, channel1);
    eth_receiver_channel_done(channel1);
    eth_wait_for_receiver_channel_done(channel0);

    buf_to_dram(write_bank, dram_end_addr, curr_recv_addr, transfer_size, recvbuf1);
    noc_barrier();

    uint64_t delta = timestamp() - start;

    *(uint64_t*)send_delta_addr = delta;
    *iter = dram_end_addr;
}
