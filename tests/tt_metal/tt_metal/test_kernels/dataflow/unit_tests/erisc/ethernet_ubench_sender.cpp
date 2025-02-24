// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet_ubench_common.hpp"

FORCE_INLINE void send_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t full_payload_size,
    uint32_t num_messages) {
    uint32_t total_msgs;
    if constexpr (measurement_type == MeasurementType::Latency) {
        total_msgs = num_messages;
    } else {
        total_msgs = num_messages * NUM_BUFFER_SLOTS;
    }

    DPRINT << "SENDER MAIN LOOP" << ENDL();

    uint32_t sender_buffer_read_ptr = 0;
    uint32_t sender_buffer_write_ptr = 0;
    uint32_t sender_num_messages_ack = 0;
    uint32_t sender_num_messages_send = 0;

    while (sender_num_messages_send < total_msgs) {
        update_sender_state(
            buffer_slot_addrs,
            full_payload_size,
            sender_num_messages_send,
            total_msgs,
            sender_buffer_read_ptr,
            sender_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }

    // wait until we got all the acks back
    while (get_ptr_val<sync_reg_id>() != NUM_BUFFER_SLOTS) {
        switch_context_if_debug();
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(is_power_of_two(NUM_BUFFER_SLOTS));

    const uint32_t full_payload_size = message_size;
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    uint32_t buffer_start_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);

    std::array<uint32_t, NUM_BUFFER_SLOTS> sender_buffer_slot_addrs;
    buffer_start_addr = setup_sender_buffer(sender_buffer_slot_addrs, buffer_start_addr, message_size);

    // Only used for bi-directional cases
    std::array<uint32_t, NUM_BUFFER_SLOTS> receiver_buffer_slot_addrs;
    if constexpr (benchmark_type == BenchmarkType::EthOnlyBiDir or benchmark_type == BenchmarkType::EthEthTensixBiDir) {
        setup_receiver_buffer(receiver_buffer_slot_addrs, buffer_start_addr, message_size);
    }

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }

    init_ptr_val<sync_reg_id>(NUM_BUFFER_SLOTS);

    eth_setup_handshake(handshake_addr, true);

    uint64_t worker_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);

    switch (benchmark_type) {
        case EthOnlyUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_uni_dir(sender_buffer_slot_addrs, full_payload_size, num_messages);
        } break;
        case EthOnlyBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<false>(
                sender_buffer_slot_addrs,
                receiver_buffer_slot_addrs,
                full_payload_size,
                message_size,
                num_messages,
                worker_noc_addr);
        } break;
        case EthEthTensixUniDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_uni_dir(sender_buffer_slot_addrs, full_payload_size, num_messages);
        } break;
        case EthEthTensixBiDir: {
            DeviceZoneScopedN("MAIN-TEST-BODY");
            send_receiver_bi_dir<true>(
                sender_buffer_slot_addrs,
                receiver_buffer_slot_addrs,
                full_payload_size,
                message_size,
                num_messages,
                worker_noc_addr);
        } break;
        case TensixPushEth: {
            ASSERT(0);
        } break;
        case EthMcastTensix: {
            ASSERT(0);
        } break;
        case EthToLocalEth: {
            ASSERT(0);
        } break;
        case EthToLocalEthAndMcastTensix: {
            ASSERT(0);
        } break;
        default: WAYPOINT("!ETH"); ASSERT(0);
    }

    // do a barrier before any noc init
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; ++i) {
        uint32_t trid = get_buffer_slot_trid(i);
        noc_async_write_barrier_with_trid(trid);
    }
    ncrisc_noc_counters_init();
}
