// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "debug/assert.h"
#include "ethernet/dataflow_api.h"
#include <array>


struct addr_sem_pair {
    uint32_t addr;
    uint32_t sem_addr;
};

static constexpr bool DISABLE_CONTEXT_SWITCHING = true;

template <bool measure>
FORCE_INLINE void roundtrip_ping(
    std::array<uint32_t, 8> channels_addrs,
    std::array<uint32_t, 8> channels_sem_addrs,
    uint32_t max_concurrent_samples,
    uint32_t page_size,
    uint32_t eth_noc_x,
    uint32_t eth_noc_y,
    uint32_t eth_channel_sync_ack_addr,
    bool is_ring_start
    ) {

    if (is_ring_start) {
        if constexpr (measure) {
            DeviceZoneScopedN("ROUNDTRIP-PING");
            for (uint32_t i = 0; i < max_concurrent_samples; i++) {
                uint32_t sender_sem = channels_sem_addrs[i];
                uint32_t buffer_addr = channels_addrs[i];
                uint64_t send_buffer_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, buffer_addr);
                uint64_t send_sem_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, sender_sem);

                noc_async_write(buffer_addr, send_buffer_noc_addr, page_size);
                noc_semaphore_inc(send_sem_noc_addr, 1);
                eth_receiver_channel_ack(i, eth_channel_sync_ack_addr);
            }

            eth_noc_async_write_barrier();

            for (uint32_t i = 0; i < max_concurrent_samples; i++) {
                if constexpr(DISABLE_CONTEXT_SWITCHING) {
                    while(!eth_bytes_are_available_on_channel(i)) {
                        asm volatile ("");
                    }
                } else {
                    eth_wait_for_bytes_on_channel(page_size, i);
                }
                eth_receiver_channel_done(i);
            }
        } else {
            for (uint32_t i = 0; i < max_concurrent_samples; i++) {
                uint32_t sender_sem = channels_sem_addrs[i];
                uint32_t buffer_addr = channels_addrs[i];
                uint64_t send_buffer_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, buffer_addr);
                uint64_t send_sem_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, sender_sem);

                noc_async_write(buffer_addr, send_buffer_noc_addr, page_size);
                noc_semaphore_inc(send_sem_noc_addr, 1);
                eth_receiver_channel_ack(i, eth_channel_sync_ack_addr);
            }

            eth_noc_async_write_barrier();

            for (uint32_t i = 0; i < max_concurrent_samples; i++) {
                if constexpr(DISABLE_CONTEXT_SWITCHING) {
                    while(!eth_bytes_are_available_on_channel(i)) {
                        asm volatile ("");
                    }
                } else {
                    eth_wait_for_bytes_on_channel(page_size, i);
                }
                eth_receiver_channel_done(i);
            }

        }
    } else {
        for (uint32_t i = 0; i < max_concurrent_samples; i++) {
            if constexpr (DISABLE_CONTEXT_SWITCHING) {
                while(!eth_bytes_are_available_on_channel(i)) {
                    asm volatile ("");
                }
            } else {
                eth_wait_for_bytes_on_channel(page_size, i);
            }
            uint32_t sender_sem = channels_sem_addrs[i];
            uint32_t buffer_addr = channels_addrs[i];
            uint64_t send_buffer_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, buffer_addr);
            uint64_t send_sem_noc_addr = get_noc_addr(eth_noc_x, eth_noc_y, sender_sem);
            noc_async_write(buffer_addr, send_buffer_noc_addr, page_size);
            noc_semaphore_inc(send_sem_noc_addr, 1);
            eth_receiver_channel_ack(i, eth_channel_sync_ack_addr);
        }

        eth_noc_async_write_barrier();

        for (uint32_t i = 0; i < max_concurrent_samples; i++) {
            eth_receiver_channel_done(i);
        }
    }
}

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
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
    const uint32_t eth_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t eth_noc_y = get_arg_val<uint32_t>(arg_idx++);
    volatile uint32_t* start_semaphore = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t init_handshake_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t init_handshake_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t init_handshake_addr = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(max_concurrent_samples <= 8);
    for (uint32_t i = 0; i < max_concurrent_samples; i++) {
        channels_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
        channels_sem_addrs[i] = get_arg_val<uint32_t>(arg_idx++);
    }

    eth_setup_handshake(handshake_addr, false);
    // We reuse the handshake address to send back acks to the sender core.
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[0] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[1] = 1;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[2] = 0;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_addr)[3] = 0;
    uint32_t eth_channel_sync_ack_addr = handshake_addr;
    // Delete me when migrating to FD2 - for now we require a worker core to act as intermediary between local chip eriscs
    // because CreateSemaphore doesn't reliably worker on ethernet cores prior to FD2
    *start_semaphore = 0;
    // Delete me when migrating to FD2
    uint64_t init_handshake_noc_addr = get_noc_addr(init_handshake_noc_x, init_handshake_noc_y, init_handshake_addr);
    // Delete me when migrating to FD2
    noc_semaphore_inc(init_handshake_noc_addr, 1);

    eth_noc_semaphore_wait(start_semaphore, 1);

    // Clear the ring
    roundtrip_ping<false>(channels_addrs, channels_sem_addrs, max_concurrent_samples, 16, eth_noc_x, eth_noc_y, eth_channel_sync_ack_addr, is_ring_start);

    {
        for (uint32_t i = 0; i < num_samples; i++) {
            roundtrip_ping<true>(channels_addrs, channels_sem_addrs, max_concurrent_samples, transfer_size, eth_noc_x, eth_noc_y, eth_channel_sync_ack_addr, is_ring_start);
        }
    }

    if (is_ring_start) {
        for (uint i = 0; i < 10000; i++) {
            DeviceZoneScopedN("ZONE-SCOPE_OVERHEAD");
        }
    }

}
