// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet_write_worker_latency_ubench_common.hpp"

static constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(0);

FORCE_INLINE uint32_t advance_buffer_slot_ptr(uint32_t curr_ptr) { return (curr_ptr + 1) % NUM_BUFFER_SLOTS; }

FORCE_INLINE void write_receiver(
    uint32_t buffer_slot_addr,
    volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr,
    uint32_t full_payload_size,
    uint32_t full_payload_size_eth_words) {
    buffer_slot_sync_addr->bytes_sent = 1;

    eth_send_bytes_over_channel_payload_only_unsafe(
        buffer_slot_addr, buffer_slot_addr, full_payload_size, full_payload_size, full_payload_size_eth_words);
}

FORCE_INLINE bool has_receiver_ack(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    return buffer_slot_sync_addr->bytes_sent == 0;
}

FORCE_INLINE void check_buffer_full_and_send_packet(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t full_payload_size,
    uint32_t full_payload_size_eth_words) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && !eth_txq_is_busy()) {
        write_receiver(
            buffer_slot_addrs[write_ptr],
            buffer_slot_sync_addrs[write_ptr],
            full_payload_size,
            full_payload_size_eth_words);

        write_ptr = next_write_ptr;
    }
}

FORCE_INLINE void check_receiver_done(
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t& read_ptr,
    uint32_t& num_messages_ack) {
    if (has_receiver_ack(buffer_slot_sync_addrs[read_ptr])) {
        read_ptr = advance_buffer_slot_ptr(read_ptr);
        num_messages_ack++;
    }
}

FORCE_INLINE void sender_main_loop(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t num_messages) {
    uint32_t full_payload_size_eth_words = full_payload_size >> 4;
    uint32_t total_msgs = num_messages * NUM_BUFFER_SLOTS;

    DPRINT << "SENDER MAIN LOOP" << ENDL();

    uint32_t buffer_read_ptr = 0;
    uint32_t buffer_write_ptr = 0;

    uint32_t num_messages_ack = 0;
    while (num_messages_ack < total_msgs) {
        // Check if current buffer slot is ready and send packet to receiver
        check_buffer_full_and_send_packet(
            buffer_slot_addrs,
            buffer_slot_sync_addrs,
            buffer_read_ptr,
            buffer_write_ptr,
            full_payload_size,
            full_payload_size_eth_words);
        // Check if the write for trid is done, and ack sender if the current buffer slot is done
        check_receiver_done(buffer_slot_sync_addrs, buffer_read_ptr, num_messages_ack);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;

    ASSERT(is_power_of_two(NUM_BUFFER_SLOTS));

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
        sender_main_loop(buffer_slot_addrs, buffer_slot_sync_addrs, full_payload_size, num_messages);
    }
}
