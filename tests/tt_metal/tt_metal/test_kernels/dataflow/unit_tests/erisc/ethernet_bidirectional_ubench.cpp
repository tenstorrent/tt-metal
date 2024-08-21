// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"

// FURTHER IMPROVEMENTS TO MAKE:
// Make reader/writer indexers increment sequentially and only advance when that
// channel index is complete. They must still increment independently.
// -> Make sure the starting indices match up between sender and receiver side

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        erisc_info->channels[0].bytes_sent = 0;
        erisc_info->channels[0].receiver_ack = 0;
        while (eth_txq_is_busy()) {}
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        while (eth_txq_is_busy()) {}
        internal_::eth_send_packet(
            0,
            ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
            ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
            1);
        while (erisc_info->channels[0].bytes_sent != 0) {}
    } else {

        while (erisc_info->channels[0].bytes_sent == 0) {
        }
        eth_receiver_channel_done(0);
    }
}

static constexpr bool MERGE_MSG = true;
static constexpr uint32_t MAX_CHANNELS = 64;
void kernel_main() {

    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_channels = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;

    ASSERT(num_channels * 2 <= 8);

    std::array<uint32_t, MAX_CHANNELS> messages_complete;
    std::array<uint32_t, MAX_CHANNELS> channel_addrs;
    std::array<volatile eth_channel_sync_t*, MAX_CHANNELS> channel_syncs;
    {
        uint32_t channel_addr = handshake_addr + sizeof(eth_channel_sync_t);
        for (uint8_t i = 0; i < num_channels*2; i++) {
            erisc_info->channels[i].bytes_sent = 0;
            erisc_info->channels[i].receiver_ack = 0;
            messages_complete[i] = 0;
            channel_addrs[i] = channel_addr;
            channel_addr += message_size;
            channel_syncs[i] = reinterpret_cast<volatile eth_channel_sync_t*>(channel_addr);
            channel_syncs[i]->bytes_sent = 0;
            channel_syncs[i]->receiver_ack = 0;
            channel_addr += sizeof(eth_channel_sync_t);
        }
    }

    const uint32_t message_size_payload = message_size + (MERGE_MSG ? sizeof(eth_channel_sync_t) : 0);
    const uint32_t message_size_payload_eth_words = message_size >> 4;

    const uint32_t last_channel = num_channels - 1;
    uint32_t channels_complete = 0;
    uint8_t senders_start = is_sender_offset_0 ? 0 : num_channels;
    uint8_t receivers_start = is_sender_offset_0 ? num_channels : 0;
    uint8_t senders_end = senders_start + num_channels;
    uint8_t receivers_end = receivers_start + num_channels;

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }
    eth_setup_handshake(handshake_addr, is_sender_offset_0);

    uint32_t ready_to_send_payload = (1 << num_channels) - 1;
    uint32_t ready_to_send_payload_available = 0;
    uint32_t wait_ack = 0;


    uint32_t idle_count = 0;
    uint32_t idle_max = 1000000000;
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        uint32_t r_i = 0;
        uint32_t s_i = 0;
        while (channels_complete < num_channels*2) {
            uint32_t sender_channel = senders_start + s_i;

            const bool sender_check_ack = (wait_ack >> s_i) & 0x1;
            if (sender_check_ack) {
                bool acked = MERGE_MSG ?
                    channel_syncs[sender_channel]->bytes_sent == 0:
                    eth_is_receiver_channel_send_done(sender_channel);
                if (acked) {
                    messages_complete[sender_channel]++;
                    wait_ack &= ~(1 << s_i);
                    if (messages_complete[sender_channel] == num_messages) {
                        channels_complete++;
                    } else {
                        ready_to_send_payload |= 1 << s_i;
                    }
                    idle_count = 0;
                }
            }
            // Sender
            const bool try_send_payload = (ready_to_send_payload >> s_i) & 0x1;
            if (try_send_payload && !eth_txq_is_busy()) {
                channel_syncs[sender_channel]->bytes_sent = message_size_payload;
                channel_syncs[sender_channel]->receiver_ack = 0;
                eth_send_bytes_over_channel_payload_only(
                    channel_addrs[sender_channel],
                    channel_addrs[sender_channel],
                    message_size_payload,
                    message_size_payload,
                    message_size_payload_eth_words + 1);
                ready_to_send_payload &= ~(1 << s_i);
                if constexpr (MERGE_MSG) {
                    wait_ack |= (1 << s_i);
                } else {
                    ready_to_send_payload_available |= (1 << s_i);
                }
                idle_count = 0;
            }

            const bool try_send_payload_available = (ready_to_send_payload_available >> s_i) & 0x1;
            if constexpr (!MERGE_MSG) {
                if (try_send_payload_available && !eth_txq_is_busy()) {

                    eth_send_payload_complete_signal_over_channel(
                        sender_channel, message_size_payload);

                    ready_to_send_payload_available &= ~(1 << s_i);
                    wait_ack |= (1 << s_i);
                    idle_count = 0;
                }
            }



            // Receiver
            uint8_t receiver_channel = receivers_start + r_i;
            if constexpr (MERGE_MSG) {
                if (channel_syncs[receiver_channel]->bytes_sent != 0) {
                    if (!eth_txq_is_busy()) {
                        channel_syncs[receiver_channel]->bytes_sent = 0;
                        channel_syncs[receiver_channel]->receiver_ack = 0;
                        internal_::eth_send_packet(
                            0,
                            ((uint32_t)channel_syncs[receiver_channel]) >> 4,
                            ((uint32_t)channel_syncs[receiver_channel]) >> 4,
                            1);
                        messages_complete[receiver_channel]++;
                        if (messages_complete[receiver_channel] == num_messages) {
                            channels_complete++;
                        }
                        idle_count = 0;
                        r_i = r_i == last_channel ? 0 : r_i + 1;
                    }
                }
            } else {
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
            }

            s_i = s_i == last_channel ? 0 : s_i + 1;

            idle_count++;
            if (idle_count > idle_max) {
                run_routing();
                idle_count = 0;
            }
        }
    }



}
