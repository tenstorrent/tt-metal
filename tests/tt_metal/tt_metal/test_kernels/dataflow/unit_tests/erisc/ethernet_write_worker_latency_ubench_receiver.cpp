// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ethernet_write_worker_latency_ubench_common.hpp"

static constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(0);
static constexpr uint32_t MAX_NUM_TRANSACTION_ID =
    NUM_BUFFER_SLOTS / 2;  // the algorithm only works for NUM_BUFFER_SLOTS divisible by MAX_NUM_TRANSACTION_ID
static constexpr uint32_t worker_noc_x = get_compile_time_arg_val(1);
static constexpr uint32_t worker_noc_y = get_compile_time_arg_val(2);
static constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(3);

FORCE_INLINE uint32_t advance_buffer_slot_ptr(uint32_t curr_ptr) { return (curr_ptr + 1) % NUM_BUFFER_SLOTS; }

FORCE_INLINE uint32_t get_buffer_slot_trid(uint32_t curr_ptr) { return curr_ptr % MAX_NUM_TRANSACTION_ID + 1; }

FORCE_INLINE bool has_incoming_packet(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    return buffer_slot_sync_addr->bytes_sent != 0;
}

FORCE_INLINE bool write_worker_done(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid);
}

FORCE_INLINE void ack_complete(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    buffer_slot_sync_addr->bytes_sent = 0;

    eth_send_bytes_over_channel_payload_only_unsafe(
        reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
        reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
        sizeof(eth_buffer_slot_sync_t),
        sizeof(eth_buffer_slot_sync_t),
        sizeof(eth_buffer_slot_sync_t) >> 4);
}

FORCE_INLINE void write_worker(
    uint32_t buffer_slot_addr,
    volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t curr_trid_to_write) {
    // write to local
    noc_async_write_one_packet_with_trid(buffer_slot_addr, worker_noc_addr, message_size, curr_trid_to_write);

    // reset sync
    buffer_slot_sync_addr->bytes_sent = 0;
}

FORCE_INLINE void check_incomping_packet_and_write_worker(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t worker_noc_addr,
    uint32_t message_size) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && has_incoming_packet(buffer_slot_sync_addrs[write_ptr])) {
        uint32_t curr_trid = get_buffer_slot_trid(write_ptr);
        write_worker(
            buffer_slot_addrs[write_ptr], buffer_slot_sync_addrs[write_ptr], worker_noc_addr, message_size, curr_trid);

        write_ptr = next_write_ptr;
    }
}

FORCE_INLINE void check_write_worker_done_and_send_ack(
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t& read_ptr,
    uint32_t write_ptr,
    uint32_t& num_messages_ack) {
    bool buffer_not_empty = read_ptr != write_ptr;
    uint32_t curr_trid = get_buffer_slot_trid(read_ptr);

    if (buffer_not_empty && write_worker_done(curr_trid) && !eth_txq_is_busy()) {
        ack_complete(buffer_slot_sync_addrs[read_ptr]);

        read_ptr = advance_buffer_slot_ptr(read_ptr);

        num_messages_ack++;
    }
}

FORCE_INLINE void receiver_main_loop(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t num_messages) {
    uint32_t total_msgs = num_messages * NUM_BUFFER_SLOTS;

    DPRINT << "RECEIVER MAIN LOOP" << ENDL();

    uint32_t buffer_read_ptr = 0;
    uint32_t buffer_write_ptr = 0;

    uint32_t num_messages_ack = 0;
    while (num_messages_ack < total_msgs) {
        // Check if there's an incoming packet for current buffer slot and write to worker if there's new packet
        check_incomping_packet_and_write_worker(
            buffer_slot_addrs,
            buffer_slot_sync_addrs,
            buffer_read_ptr,
            buffer_write_ptr,
            worker_noc_addr,
            message_size);
        // Check if the write for trid is done, and ack sender if the current buffer slot is done
        check_write_worker_done_and_send_ack(
            buffer_slot_sync_addrs, buffer_read_ptr, buffer_write_ptr, num_messages_ack);

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

    std::array<uint32_t, NUM_BUFFER_SLOTS> buffer_slot_addrs;
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS> buffer_slot_sync_addrs;
    {
        uint32_t buffer_slot_addr = handshake_addr + sizeof(eth_buffer_slot_sync_t);
        for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
            buffer_slot_addrs[i] = buffer_slot_addr;
            buffer_slot_addr += message_size;
            buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(buffer_slot_addr);
            buffer_slot_sync_addrs[i]->bytes_sent = 0;
            buffer_slot_sync_addrs[i]->receiver_ack = 0;
            buffer_slot_addr += sizeof(eth_buffer_slot_sync_t);
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
        receiver_main_loop(buffer_slot_addrs, buffer_slot_sync_addrs, worker_noc_addr, message_size, num_messages);
    }
}
