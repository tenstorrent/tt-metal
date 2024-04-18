// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t MAX_CHANNELS = 8;
void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_channels = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t message_size_eth_words = message_size >> 4;

    ASSERT(num_channels * 2 <= 8);

    std::array<uint32_t, MAX_CHANNELS> messages_complete;
    std::array<uint32_t, MAX_CHANNELS> channel_addrs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < num_channels*2; i++) {
            messages_complete[i] = 0;
            channel_addrs[i] = channel_addr;
            channel_addr += message_size;
        }
    }

    const uint32_t last_channel = num_channels - 1;
    uint32_t channels_complete = 0;
    uint8_t senders_start = is_sender_offset_0 ? 0 : num_channels;
    uint8_t receivers_start = is_sender_offset_0 ? num_channels : 0;
    uint8_t senders_end = senders_start + num_channels;
    uint8_t receivers_end = receivers_start + num_channels;

    eth_setup_handshake(handshake_addr, is_sender_offset_0);

    uint32_t ready_to_send_payload = (1 << num_channels) - 1;
    uint32_t ready_to_send_payload_available = 0;
    uint32_t wait_ack = 0;

    uint32_t idle_count = 0;
    uint32_t idle_max = 1000000000;
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        uint32_t i = 0;
        while (channels_complete < num_channels*2) {
            uint32_t sender_channel = senders_start + i;

            // Sender
            const bool try_send_payload = (ready_to_send_payload >> i) & 0x1;
            if (try_send_payload && !eth_txq_is_busy()) {
                eth_send_bytes_over_channel_payload_only(
                    channel_addrs[sender_channel],
                    channel_addrs[sender_channel],
                    message_size,
                    sender_channel,
                    message_size,
                    message_size_eth_words);
                ready_to_send_payload &= ~(1 << i);
                ready_to_send_payload_available |= (1 << i);
                idle_count = 0;
            }

            const bool try_send_payload_available = (ready_to_send_payload_available >> i) & 0x1;
            if (try_send_payload_available && !eth_txq_is_busy()) {
                eth_send_payload_complete_signal_over_channel(
                    sender_channel, message_size);
                ready_to_send_payload_available &= ~(1 << i);
                wait_ack |= (1 << i);
                idle_count = 0;
            }

            const bool sender_check_ack = (wait_ack >> i) & 0x1;
            if (sender_check_ack) {
                bool acked = eth_is_receiver_channel_send_done(sender_channel);
                if (acked) {
                    messages_complete[sender_channel]++;
                    wait_ack &= ~(1 << i);
                    if (messages_complete[sender_channel] == num_messages) {
                        channels_complete++;
                    } else {
                        ready_to_send_payload |= 1 << i;
                    }
                    idle_count = 0;
                }
            }

            // Receiver
            uint8_t receiver_channel = receivers_start + i;
            if (eth_bytes_are_available_on_channel(receiver_channel)) {
                if (!eth_txq_is_busy()) {
                    eth_receiver_channel_done(receiver_channel);
                    messages_complete[receiver_channel]++;
                    if (messages_complete[receiver_channel] == num_messages) {
                        channels_complete++;
                    }
                    idle_count = 0;
                }
            }

            i = i == last_channel ? 0 : i + 1;

            idle_count++;
            if (idle_count > idle_max) {
                run_routing();
                idle_count = 0;
            }
        }
    }



}
