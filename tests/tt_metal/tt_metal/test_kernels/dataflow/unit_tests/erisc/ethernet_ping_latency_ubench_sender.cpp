// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"
#include "debug/debug.h"

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

template <bool MEASURE>
FORCE_INLINE void run_loop_iteration(
    std::array<uint32_t, NUM_CHANNELS> const& channel_addrs,
    std::array<volatile eth_channel_sync_t *, NUM_CHANNELS> const& channel_sync_addrs,
    uint32_t full_payload_size,
    uint32_t full_payload_size_eth_words
) {
    if constexpr (MEASURE) {
        DeviceZoneScopedN("SENDER-LOOP-ITER");
        {
            DeviceZoneScopedN("SEND-PAYLOADS-PHASE");
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                channel_sync_addrs[i]->bytes_sent = 1;
                channel_sync_addrs[i]->receiver_ack = 0;
                eth_send_bytes_over_channel_payload_only(
                    channel_addrs[i],
                    channel_addrs[i],
                    full_payload_size,
                    full_payload_size,
                    full_payload_size_eth_words);
            }
        }
        {
            DeviceZoneScopedN("WAIT-ACKS-PHASE");
            for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
                while (channel_sync_addrs[i]->bytes_sent != 0) { }
            }
        }
    } else {

        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            channel_sync_addrs[i]->bytes_sent = 1;
            channel_sync_addrs[i]->receiver_ack = 0;
            eth_send_bytes_over_channel_payload_only(
                channel_addrs[i],
                channel_addrs[i],
                full_payload_size,
                full_payload_size,
                full_payload_size_eth_words);
        }
        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            while (channel_sync_addrs[i]->bytes_sent != 0) { }
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
    const uint32_t full_payload_size_eth_words =full_payload_size >> 4;

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

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }
    eth_setup_handshake(handshake_addr, true);

    run_loop_iteration<false>(
        channel_addrs,
        channel_sync_addrs,
        full_payload_size,
        full_payload_size_eth_words
    );
    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        uint32_t i = 0;
        for (uint32_t i = 0; i < num_messages; i++) {

            while (eth_txq_is_busy()) {
                // Start on an empty q (don't let separate loop iterations interfere with each other)
            }

            run_loop_iteration<true>(
                channel_addrs,
                channel_sync_addrs,
                full_payload_size,
                full_payload_size_eth_words
            );
        }
    }
}
