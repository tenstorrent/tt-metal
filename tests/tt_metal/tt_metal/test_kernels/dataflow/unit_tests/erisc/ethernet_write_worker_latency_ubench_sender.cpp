// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "debug/debug.h"

#define ENABLE_DEBUG 1

struct eth_buffer_slot_sync_t {
    volatile uint32_t bytes_sent;
    volatile uint32_t receiver_ack;
    volatile uint32_t src_id;

    uint32_t reserved_2;
};

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(0);

FORCE_INLINE void wait_ack(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    while (buffer_slot_sync_addr->bytes_sent != 0) {
#if ENABLE_DEBUG
        internal_::risc_context_switch();
#endif
    }
}

FORCE_INLINE void wait_eth_txq() {
    while (eth_txq_is_busy()) {
#if ENABLE_DEBUG
        internal_::risc_context_switch();
#endif
    }
}

template <bool waiting_tails>
FORCE_INLINE void run_loop_iteration(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t full_payload_size_eth_words) {
    if constexpr (!waiting_tails) {
        // send packet
        for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
            // wait for receiver ack compelete
            wait_ack(buffer_slot_sync_addrs[i]);

            buffer_slot_sync_addrs[i]->bytes_sent = 1;

            wait_eth_txq();

#if ENABLE_DEBUG
            eth_send_bytes_over_channel_payload_only(
#else
            eth_send_bytes_over_channel_payload_only_unsafe(
#endif
                buffer_slot_addrs[i],
                buffer_slot_addrs[i],
                full_payload_size,
                full_payload_size,
                full_payload_size_eth_words);
        }
    } else {
        // waiting tails
        for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
            // wait for receiver ack compelete
            wait_ack(buffer_slot_sync_addrs[i]);
        }
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;

    const uint32_t message_size_eth_words = message_size >> 4;

    const uint32_t full_payload_size = message_size + sizeof(eth_buffer_slot_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    std::array<uint32_t, NUM_BUFFER_SLOTS> buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> buffer_slot_sync_addrs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);
        for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
            buffer_slot_addrs[i] = channel_addr;
            channel_addr += message_size;
            buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(channel_addr);
            channel_addr += sizeof(eth_buffer_slot_sync_t);
        }
    }

    // reset bytes_sent to 0s so first iter it won't block
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_sync_addrs[i]->bytes_sent = 0;
    }

    // assemble a packet filled with values
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        tt_l1_ptr uint8_t* ptr = reinterpret_cast<tt_l1_ptr uint8_t*>(buffer_slot_addrs[i]);
        for (uint32_t j = 0; j < message_size; j++) {
            ptr[j] = j;
        }
    }

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }
    eth_setup_handshake(handshake_addr, true);

    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        for (uint32_t i = 0; i < num_messages; i++) {
            run_loop_iteration<false>(
                buffer_slot_addrs, buffer_slot_sync_addrs, full_payload_size, full_payload_size_eth_words);
        }
        run_loop_iteration<true>(
            buffer_slot_addrs, buffer_slot_sync_addrs, full_payload_size, full_payload_size_eth_words);
    }
}
