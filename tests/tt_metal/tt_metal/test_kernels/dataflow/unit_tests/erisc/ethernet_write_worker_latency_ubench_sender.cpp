// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet_write_worker_latency_ubench_common.hpp"

FORCE_INLINE void main_loop_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t num_messages) {
    uint32_t full_payload_size_eth_words = full_payload_size >> 4;
    uint32_t total_msgs = num_messages * NUM_BUFFER_SLOTS;

    DPRINT << "SENDER MAIN LOOP" << ENDL();

    uint32_t sender_buffer_read_ptr = 0;
    uint32_t sender_buffer_write_ptr = 0;
    uint32_t sender_num_messages_ack = 0;

    while (sender_num_messages_ack < total_msgs) {
        update_sender_state(
            buffer_slot_addrs,
            buffer_slot_sync_addrs,
            full_payload_size,
            full_payload_size_eth_words,
            sender_num_messages_ack,
            sender_buffer_read_ptr,
            sender_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}

FORCE_INLINE void main_loop_bi_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& sender_buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& sender_buffer_slot_sync_addrs,
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& receiver_buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t message_size,
    uint32_t num_messages,
    uint64_t worker_noc_addr) {
    uint32_t full_payload_size_eth_words = full_payload_size >> 4;
    uint32_t total_msgs = num_messages * NUM_BUFFER_SLOTS;

    DPRINT << "SENDER MAIN LOOP" << ENDL();

    uint32_t sender_buffer_read_ptr = 0;
    uint32_t sender_buffer_write_ptr = 0;
    uint32_t sender_num_messages_ack = 0;

    uint32_t receiver_buffer_read_ptr = 0;
    uint32_t receiver_buffer_write_ptr = 0;
    uint32_t receiver_num_messages_ack = 0;

    while (sender_num_messages_ack < total_msgs || receiver_num_messages_ack < total_msgs) {
        if (sender_num_messages_ack < total_msgs) {
            update_sender_state(
                sender_buffer_slot_addrs,
                sender_buffer_slot_sync_addrs,
                full_payload_size,
                full_payload_size_eth_words,
                sender_num_messages_ack,
                sender_buffer_read_ptr,
                sender_buffer_write_ptr);
        }

        if (receiver_num_messages_ack < total_msgs) {
            update_receiver_state(
                receiver_buffer_slot_addrs,
                receiver_buffer_slot_sync_addrs,
                worker_noc_addr,
                message_size,
                receiver_num_messages_ack,
                receiver_buffer_read_ptr,
                receiver_buffer_write_ptr);
        }

        // not called in normal execution mode
        switch_context_if_debug();
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);

    ASSERT(is_power_of_two(NUM_BUFFER_SLOTS));

    const uint32_t full_payload_size = message_size + sizeof(eth_buffer_slot_sync_t);
    const uint32_t full_payload_size_eth_words = full_payload_size >> 4;

    uint32_t buffer_start_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);

    std::array<uint32_t, NUM_BUFFER_SLOTS> sender_buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> sender_buffer_slot_sync_addrs;
    buffer_start_addr =
        setup_sender_buffer(sender_buffer_slot_addrs, sender_buffer_slot_sync_addrs, buffer_start_addr, message_size);

#ifdef ENABLE_BI_DIRECTION
    std::array<uint32_t, NUM_BUFFER_SLOTS> receiver_buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> receiver_buffer_slot_sync_addrs;
    setup_receiver_buffer(receiver_buffer_slot_addrs, receiver_buffer_slot_sync_addrs, buffer_start_addr, message_size);
#endif

    // Avoids hang in issue https://github.com/tenstorrent/tt-metal/issues/9963
    for (uint32_t i = 0; i < 2000000000; i++) {
        asm volatile("nop");
    }
    eth_setup_handshake(handshake_addr, true);

    // worker noc address
#ifdef ENABLE_BI_DIRECTION
    uint64_t worker_noc_addr = get_noc_addr(worker_noc_x, worker_noc_y, worker_buffer_addr);
#endif

    {
        DeviceZoneScopedN("MAIN-TEST-BODY");
#ifdef ENABLE_BI_DIRECTION
        main_loop_bi_dir(
            sender_buffer_slot_addrs,
            sender_buffer_slot_sync_addrs,
            receiver_buffer_slot_addrs,
            receiver_buffer_slot_sync_addrs,
            full_payload_size,
            message_size,
            num_messages,
            worker_noc_addr);
#else
        main_loop_uni_dir(sender_buffer_slot_addrs, sender_buffer_slot_sync_addrs, full_payload_size, num_messages);
#endif
    }
}
