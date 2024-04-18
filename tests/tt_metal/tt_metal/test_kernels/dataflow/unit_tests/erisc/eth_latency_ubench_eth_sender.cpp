// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dataflow_api.h"
#include "debug/assert.h"
#include <array>
#include "ethernet/dataflow_api.h"


struct addr_sem_pair {
    uint32_t addr;
    uint32_t sem_addr;
};

static constexpr bool DISABLE_CONTEXT_SWITCHING = true;

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

template <bool measure>
FORCE_INLINE void forward_ping(std::array<uint32_t, 8> const& channels_addrs, std::array<uint32_t, 8> const& channels_sem_addrs, uint32_t max_concurrent_samples, uint32_t page_size) {

    for (uint32_t i = 0; i < max_concurrent_samples; i++) {
        volatile uint32_t *sem_addr = reinterpret_cast<volatile uint32_t*>(channels_sem_addrs[i]);
        uint32_t buffer_addr = channels_addrs[i];
        if constexpr (DISABLE_CONTEXT_SWITCHING) {
            noc_semaphore_wait(sem_addr, 1);
        } else {
            eth_noc_semaphore_wait(sem_addr, 1);
        }
        *sem_addr = 0;
        eth_send_bytes_over_channel(buffer_addr, buffer_addr, page_size, i);
    }
    for (uint32_t i = 0; i < max_concurrent_samples; i++) {
        if constexpr (DISABLE_CONTEXT_SWITCHING) {
            while (!eth_is_receiver_channel_send_done(i)) {
                asm volatile ("");
            }
        } else {
            eth_wait_for_receiver_channel_done(i);
        }
    }
}


void kernel_main() {
    std::array<uint32_t, 8> channels_addrs;
    std::array<uint32_t, 8> channels_sem_addrs;
    uint32_t arg_idx = 0;
    const bool is_ring_start = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_samples = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t max_concurrent_samples = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t transfer_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_receiver_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t local_receiver_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t receiver_start_semaphore = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(max_concurrent_samples <= 8);
    for (uint32_t i = 0; i < max_concurrent_samples; i++) {
        channels_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
        channels_sem_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    for (uint32_t i = 0; i < max_concurrent_samples; i++) {
        *(volatile uint32_t*)channels_sem_addrs[i] = 0;
    }

    {
    eth_setup_handshake(handshake_addr, true);
    }

    uint64_t receiver_start_semaphore_noc_addr = get_noc_addr(local_receiver_noc_x, local_receiver_noc_y, receiver_start_semaphore);
    noc_semaphore_inc(receiver_start_semaphore_noc_addr, 1);

    // Clear the ring
    forward_ping<false>(channels_addrs, channels_sem_addrs, max_concurrent_samples, 16);

    for (uint32_t i = 0; i < num_samples; i++) {
        forward_ping<true>(channels_addrs, channels_sem_addrs, max_concurrent_samples, transfer_size);
    }

}
