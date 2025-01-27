// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "eth_l1_address_map.h"
#include "dataflow_api.h"
#include "ethernet/dataflow_api.h"
#include "debug/assert.h"
#include "debug/dprint.h"

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

static constexpr uint32_t MAX_TRANSACTION_ID = 15;
static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);
static constexpr uint32_t worker_noc_x = get_compile_time_arg_val(1);
static constexpr uint32_t worker_noc_y = get_compile_time_arg_val(2);
static constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(3);
static constexpr bool use_transaction_id = get_compile_time_arg_val(4) == 1;
static constexpr uint32_t num_writes_skip_barrier = get_compile_time_arg_val(5);

template <bool sending_tails>
FORCE_INLINE void run_loop_iteration(
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size) {
    if constexpr (!sending_tails) {
        static uint32_t writes_count;

        for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
            if constexpr (use_transaction_id) {
                uint32_t curr_transaction_id = i % MAX_TRANSACTION_ID + 1;

                if (writes_count < num_writes_skip_barrier) {
                    // wait for sender data arrive
                    while (channel_sync_addrs[i]->bytes_sent == 0) {
                    }

                    noc_async_write_with_trid(channel_addrs[i], worker_noc_addr, message_size, curr_transaction_id);

                    DPRINT << "write CH: " << i << " tid: " << curr_transaction_id << ENDL();

                    // not using any barrier
                    writes_count++;
                } else {
                    uint32_t prev_channel_id_to_ack = (NUM_CHANNELS - num_writes_skip_barrier + i) % NUM_CHANNELS;
                    uint32_t prev_transaction_id = prev_channel_id_to_ack % MAX_TRANSACTION_ID + 1;

                    // barrier on previous data
                    noc_async_write_barrier_with_trid(prev_transaction_id);

                    channel_sync_addrs[prev_channel_id_to_ack]->bytes_sent = 0;
                    channel_sync_addrs[prev_channel_id_to_ack]->receiver_ack = 0;

                    // wait for txq to be ready, otherwise we'll
                    // hit a context switch in the send command
                    while (eth_txq_is_busy()) {
                    }
                    eth_send_bytes_over_channel_payload_only_unsafe(
                        reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
                        reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
                        sizeof(eth_channel_sync_t),
                        sizeof(eth_channel_sync_t),
                        sizeof(eth_channel_sync_t) >> 4);

                    DPRINT << "ack CH: " << prev_channel_id_to_ack << ENDL();
                    DPRINT << "wait tid: " << prev_transaction_id << ENDL();

                    // wait only after the ack is sent for the previous packet
                    while (channel_sync_addrs[i]->bytes_sent == 0) {
                    }

                    noc_async_write_with_trid(channel_addrs[i], worker_noc_addr, message_size, curr_transaction_id);

                    DPRINT << "write CH: " << i << " tid: " << curr_transaction_id << ENDL();
                }
            } else {
                // wait for sender data arrive
                while (channel_sync_addrs[i]->bytes_sent == 0) {
                }

                channel_sync_addrs[i]->bytes_sent = 0;
                channel_sync_addrs[i]->receiver_ack = 0;

                // for current channel data, send it to worker core and perform barrier
                noc_async_write(channel_addrs[i], worker_noc_addr, message_size);
                noc_async_write_barrier();

                // wait for txq to be ready, otherwise we'll
                // hit a context switch in the send command
                while (eth_txq_is_busy()) {
                }
                eth_send_bytes_over_channel_payload_only_unsafe(
                    reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                    reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
                    sizeof(eth_channel_sync_t),
                    sizeof(eth_channel_sync_t),
                    sizeof(eth_channel_sync_t) >> 4);
            }
        }
    } else {
        // sending tails
        for (uint32_t i = 0; i < num_writes_skip_barrier; i++) {
            // barrier on previous data
            uint32_t prev_channel_id_to_ack = NUM_CHANNELS - num_writes_skip_barrier + i;
            uint32_t prev_transaction_id = prev_channel_id_to_ack % MAX_TRANSACTION_ID + 1;

            noc_async_write_barrier_with_trid(prev_transaction_id);

            DPRINT << "tails wait tid: " << prev_transaction_id << ENDL();
            DPRINT << "tails ack CH: " << prev_channel_id_to_ack << ENDL();

            channel_sync_addrs[prev_channel_id_to_ack]->bytes_sent = 0;
            channel_sync_addrs[prev_channel_id_to_ack]->receiver_ack = 0;

            // wait for txq to be ready, otherwise we'll
            // hit a context switch in the send command
            while (eth_txq_is_busy()) {
            }
            eth_send_bytes_over_channel_payload_only_unsafe(
                reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
                reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t),
                sizeof(eth_channel_sync_t) >> 4);
        }
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(num_writes_skip_barrier <= NUM_CHANNELS);

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

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }

    // worker noc address
    uint64_t worker_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);

    eth_setup_handshake(handshake_addr, false);

    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
        for (uint32_t i = 0; i < num_messages; i++) {
            run_loop_iteration<false>(channel_addrs, channel_sync_addrs, worker_noc_addr, message_size);
        }

        if constexpr (use_transaction_id) {
            run_loop_iteration<true>(channel_addrs, channel_sync_addrs, worker_noc_addr, message_size);
        }
    }
}
