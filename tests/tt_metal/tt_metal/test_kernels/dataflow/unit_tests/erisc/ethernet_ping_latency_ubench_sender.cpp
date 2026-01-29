// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);

// Stream IDs for credit management (using unused stream registers)
static constexpr uint32_t SENDER_CREDIT_STREAM_ID = 0;
static constexpr uint32_t TXQ_ID = 0;

template <bool MEASURE>
FORCE_INLINE void run_loop_iteration(
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
    uint32_t full_payload_size,
    [[maybe_unused]] uint32_t full_payload_size_eth_words,
    int32_t& expected_credits) {
    if constexpr (MEASURE) {
        DeviceZoneScopedN("SENDER-LOOP-ITER");
        {
            DeviceZoneScopedN("SEND-PAYLOADS-PHASE");
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                // Set credit flag before sending data
                channel_sync_addrs[i]->bytes_sent = 1;
                channel_sync_addrs[i]->receiver_ack = 0;
                // Wait for txq to be ready before sending (outside the API call for better performance)
                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                // Send data packet using unsafe API (assumes txq is ready)
                internal_::eth_send_packet_bytes_unsafe(TXQ_ID, channel_addrs[i], channel_addrs[i], full_payload_size);
                // Send credit update to receiver via stream register
                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                remote_update_ptr_val<SENDER_CREDIT_STREAM_ID, TXQ_ID>(1);
            }
        }
        {
            DeviceZoneScopedN("WAIT-ACKS-PHASE");
            // Wait for receiver to send credits back via stream register
            while (get_ptr_val<SENDER_CREDIT_STREAM_ID>() < expected_credits) {
            }
            // Consume the credits
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                increment_local_update_ptr_val<SENDER_CREDIT_STREAM_ID>(-1);
                expected_credits--;
            }
        }
    } else {
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            // Set credit flag before sending data
            channel_sync_addrs[i]->bytes_sent = 1;
            channel_sync_addrs[i]->receiver_ack = 0;
            while (internal_::eth_txq_is_busy(TXQ_ID)) {
            }
            // Send data packet using unsafe API (assumes txq is ready)
            internal_::eth_send_packet_bytes_unsafe(TXQ_ID, channel_addrs[i], channel_addrs[i], full_payload_size);
            // Send credit update to receiver via stream register
            while (internal_::eth_txq_is_busy(TXQ_ID)) {
            }
            remote_update_ptr_val<SENDER_CREDIT_STREAM_ID, TXQ_ID>(1);
        }
        // Wait for receiver to send credits back via stream register
        while (get_ptr_val<SENDER_CREDIT_STREAM_ID>() < expected_credits) {
        }
        // Consume the credits
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            increment_local_update_ptr_val<SENDER_CREDIT_STREAM_ID>(-1);
            expected_credits--;
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

    const uint32_t full_payload_size = message_size + sizeof(eth_channel_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    ASSERT(NUM_CHANNELS * 2 <= 8);

    std::array<uint32_t, NUM_CHANNELS> channel_addrs;
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> channel_sync_addrs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            channel_addrs[i] = channel_addr;
            channel_addr += message_size;
            channel_sync_addrs[i] = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
            channel_addr += sizeof(eth_channel_sync_t);
        }
    }

    // Initialize stream registers for credit management before handshake
    // This ensures registers are initialized before the remote side can access them
    init_ptr_val<SENDER_CREDIT_STREAM_ID>(0);

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }
    eth_setup_handshake(handshake_addr, true);

    // Track expected credits for stream register checking
    int32_t expected_credits = NUM_CHANNELS;

    run_loop_iteration<false>(
        channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words, expected_credits);
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        for (uint32_t i = 0; i < num_messages; i++) {
            while (eth_txq_is_busy()) {
                // Start on an empty q (don't let separate loop iterations interfere with each other)
            }

            expected_credits += NUM_CHANNELS;
            run_loop_iteration<true>(
                channel_addrs, channel_sync_addrs, full_payload_size, full_payload_size_eth_words, expected_credits);
        }
    }
}
