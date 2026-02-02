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
static constexpr uint32_t RECEIVER_CREDIT_STREAM_ID = 0;
static constexpr uint32_t TXQ_ID = 0;

template <bool MEASURE>
FORCE_INLINE void run_loop_iteration(
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
    int32_t& expected_credits,
    uint32_t full_payload_size) {
    if constexpr (MEASURE) {
        DeviceZoneScopedN("RECEIVER-LOOP-ITER");
        // Wait for credit from sender (check stream register)
        while (get_ptr_val<RECEIVER_CREDIT_STREAM_ID>() < expected_credits) {
        }

        {
            DeviceZoneScopedN("PING-REPLIES");
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                // Consume the credit
                increment_local_update_ptr_val<RECEIVER_CREDIT_STREAM_ID>(-1);
                expected_credits--;

                // Clear flags and send credit back to sender
                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                // Wait for txq to be ready before sending (outside the API call for better performance)
                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                // Send back same sized packet (echo the full payload)
                internal_::eth_send_packet_bytes_unsafe(TXQ_ID, channel_addrs[i], channel_addrs[i], full_payload_size);
                // Send credit update to sender via stream register
                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                remote_update_ptr_val<RECEIVER_CREDIT_STREAM_ID, TXQ_ID>(1);
            }
        }
    } else {
        // Wait for credit from sender (check stream register)
        while (get_ptr_val<RECEIVER_CREDIT_STREAM_ID>() < expected_credits) {
        }

        {
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                // Consume the credit
                increment_local_update_ptr_val<RECEIVER_CREDIT_STREAM_ID>(-1);
                expected_credits--;

                // Clear flags and send credit back to sender
                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                // Send back same sized packet (echo the full payload)
                internal_::eth_send_packet_bytes_unsafe(TXQ_ID, channel_addrs[i], channel_addrs[i], full_payload_size);
                // Send credit update to sender via stream register
                while (internal_::eth_txq_is_busy(TXQ_ID)) {
                }
                remote_update_ptr_val<RECEIVER_CREDIT_STREAM_ID, TXQ_ID>(1);
            }
        }
    }
}

static constexpr uint32_t MAX_CHANNELS = 8;
void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    std::array<uint32_t, NUM_CHANNELS> channel_addrs;
    std::array<volatile eth_channel_sync_t*, NUM_CHANNELS> channel_sync_addrs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < NUM_CHANNELS; i++) {
            channel_addrs[i] = channel_addr;
            channel_addr += message_size;
            channel_sync_addrs[i] = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
            channel_sync_addrs[i]->bytes_sent = 0;
            channel_sync_addrs[i]->receiver_ack = 0;
            channel_addr += sizeof(eth_channel_sync_t);
        }
    }

    // Initialize stream registers for credit management before handshake
    // This ensures registers are initialized before the remote side can access them
    init_ptr_val<RECEIVER_CREDIT_STREAM_ID>(0);

    eth_setup_handshake(handshake_addr, false);

    // Calculate full payload size (same as sender)
    const uint32_t full_payload_size = message_size + sizeof(eth_channel_sync_t);

    // Track expected credits for stream register checking
    int32_t expected_credits = NUM_CHANNELS;

    run_loop_iteration<false>(channel_addrs, channel_sync_addrs, expected_credits, full_payload_size);
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        for (uint32_t i = 0; i < num_messages; i++) {
            expected_credits += NUM_CHANNELS;
            run_loop_iteration<true>(channel_addrs, channel_sync_addrs, expected_credits, full_payload_size);
        }
    }
}
