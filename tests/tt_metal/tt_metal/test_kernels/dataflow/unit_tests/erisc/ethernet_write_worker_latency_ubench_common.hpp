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
#include "eth_ubenchmark_types.hpp"
#include "risc_common.h"

// #define ENABLE_DEBUG 1

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_channel_done(0);
    }
}

FORCE_INLINE void switch_context_if_debug() {
#if ENABLE_DEBUG
    internal_::risc_context_switch();
#endif
}

template <typename T>
bool is_power_of_two(T val) {
    return (val & (val - 1)) == T(0);
}

// ******************************* Common Ct Args ************************************************

constexpr BenchmarkType benchmark_type = static_cast<BenchmarkType>(get_compile_time_arg_val(0));
constexpr MeasurementType measurement_type = static_cast<MeasurementType>(get_compile_time_arg_val(1));
constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(2);
constexpr uint32_t MAX_NUM_TRANSACTION_ID =
    NUM_BUFFER_SLOTS / 2;  // the algorithm only works for NUM_BUFFER_SLOTS divisible by MAX_NUM_TRANSACTION_ID
constexpr uint32_t disable_trid = get_compile_time_arg_val(3);

// ******************************* Sender APIs ***************************************************

FORCE_INLINE uint32_t setup_sender_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t buffer_slot_addr,
    uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
        buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(buffer_slot_addr);
        buffer_slot_addr += sizeof(eth_buffer_slot_sync_t);
    }

    // reset bytes_sent to 1s so first iter it will block on receiver ack
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_sync_addrs[i]->bytes_sent = 1;
    }

    // assemble a packet filled with values
    for (uint32_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        tt_l1_ptr uint8_t* ptr = reinterpret_cast<tt_l1_ptr uint8_t*>(buffer_slot_addrs[i]);
        for (uint32_t j = 0; j < message_size; j++) {
            ptr[j] = j;
        }
    }

    uint32_t buffer_end_addr = buffer_slot_addr;
    return buffer_end_addr;
}

FORCE_INLINE uint32_t advance_buffer_slot_ptr(uint32_t curr_ptr) { return (curr_ptr + 1) % NUM_BUFFER_SLOTS; }

FORCE_INLINE void write_receiver(
    uint32_t buffer_slot_addr, volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr, uint32_t full_payload_size) {
    buffer_slot_sync_addr->bytes_sent = 1;

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }

    eth_send_bytes_over_channel_payload_only_unsafe_one_packet(buffer_slot_addr, buffer_slot_addr, full_payload_size);
}

FORCE_INLINE bool has_receiver_ack(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    invalidate_l1_cache();
    return buffer_slot_sync_addr->bytes_sent == 0;
}

FORCE_INLINE void check_buffer_full_and_send_packet(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t full_payload_size,
    uint32_t& num_messages_send) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && num_messages_send != 0) {
        write_receiver(buffer_slot_addrs[write_ptr], buffer_slot_sync_addrs[write_ptr], full_payload_size);

        write_ptr = next_write_ptr;
        num_messages_send--;
    }
}

FORCE_INLINE void check_receiver_done(
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t& read_ptr,
    uint32_t& num_messages_ack) {
    if (has_receiver_ack(buffer_slot_sync_addrs[read_ptr])) {
        uint32_t next_read_ptr = advance_buffer_slot_ptr(read_ptr);

        buffer_slot_sync_addrs[read_ptr]->bytes_sent = 1;
        read_ptr = next_read_ptr;
        num_messages_ack++;
    }
}

FORCE_INLINE void update_sender_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t& num_messages_ack,
    uint32_t& num_messages_send,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check if current buffer slot is ready and send packet to receiver
    check_buffer_full_and_send_packet(
        buffer_slot_addrs,
        buffer_slot_sync_addrs,
        buffer_read_ptr,
        buffer_write_ptr,
        full_payload_size,
        num_messages_send);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_receiver_done(buffer_slot_sync_addrs, buffer_read_ptr, num_messages_ack);
}

// ******************************* Receiver APIs *************************************************

FORCE_INLINE uint32_t setup_receiver_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t buffer_slot_addr,
    uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
        buffer_slot_sync_addrs[i] = reinterpret_cast<volatile eth_buffer_slot_sync_t*>(buffer_slot_addr);
        buffer_slot_sync_addrs[i]->bytes_sent = 0;
        buffer_slot_sync_addrs[i]->receiver_ack = 0;
        buffer_slot_addr += sizeof(eth_buffer_slot_sync_t);
    }

    uint32_t buffer_end_addr = buffer_slot_addr;
    return buffer_end_addr;
}

FORCE_INLINE uint32_t get_buffer_slot_trid(uint32_t curr_ptr) { return curr_ptr % MAX_NUM_TRANSACTION_ID + 1; }

FORCE_INLINE bool has_incoming_packet(volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr) {
    invalidate_l1_cache();
    return buffer_slot_sync_addr->bytes_sent != 0;
}

FORCE_INLINE bool write_worker_done(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, trid);
}

FORCE_INLINE void ack_complete(
    uint32_t buffer_slot_addr, volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr, uint32_t full_payload_size) {
    buffer_slot_sync_addr->bytes_sent = 0;

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }

    if constexpr (measurement_type == MeasurementType::Latency) {
        // Send pack entire packet so measurement from sender -> receiver -> sender is symmetric
        eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
            buffer_slot_addr, buffer_slot_addr, full_payload_size);
    } else {
        eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
            reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
            reinterpret_cast<uint32_t>(buffer_slot_sync_addr),
            sizeof(eth_buffer_slot_sync_t));
    }
}

FORCE_INLINE void write_worker(
    uint32_t buffer_slot_addr,
    volatile eth_buffer_slot_sync_t* buffer_slot_sync_addr,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t curr_trid_to_write) {
    // write to local
#ifdef DISABLE_TRID
    noc_async_write(buffer_slot_addr, worker_noc_addr, message_size);
    noc_async_writes_flushed();
#else
    noc_async_write_one_packet_with_trid_with_state<false, false>(
        buffer_slot_addr, worker_noc_addr, message_size, curr_trid_to_write);
#endif
    // reset sync
    buffer_slot_sync_addr->bytes_sent = 0;
}

template <bool write_to_worker>
FORCE_INLINE void check_incoming_packet_and_write_worker(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t worker_noc_addr,
    uint32_t message_size) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && has_incoming_packet(buffer_slot_sync_addrs[write_ptr])) {
        if constexpr (write_to_worker) {
            uint32_t curr_trid = get_buffer_slot_trid(write_ptr);
            write_worker(
                buffer_slot_addrs[write_ptr],
                buffer_slot_sync_addrs[write_ptr],
                worker_noc_addr,
                message_size,
                curr_trid);
        }
        write_ptr = next_write_ptr;
    }
}

template <bool write_to_worker>
FORCE_INLINE void check_write_worker_done_and_send_ack(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t& read_ptr,
    uint32_t write_ptr,
    uint32_t& num_messages_ack) {
    bool buffer_not_empty = read_ptr != write_ptr;

    bool send_ack_condition = buffer_not_empty;
    if constexpr (write_to_worker and !disable_trid) {
        uint32_t curr_trid = get_buffer_slot_trid(read_ptr);
        send_ack_condition = send_ack_condition && write_worker_done(curr_trid);
    }
    if (send_ack_condition) {
        // DPRINT << "read_ptr " << read_ptr << ENDL();
        ack_complete(buffer_slot_addrs[read_ptr], buffer_slot_sync_addrs[read_ptr], full_payload_size);
        read_ptr = advance_buffer_slot_ptr(read_ptr);
        num_messages_ack++;
    }
}

template <bool write_to_worker>
FORCE_INLINE void update_receiver_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& buffer_slot_sync_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t full_payload_size,
    uint32_t& num_messages_ack,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check if there's an incoming packet for current buffer slot and write to worker if there's new packet
    check_incoming_packet_and_write_worker<write_to_worker>(
        buffer_slot_addrs, buffer_slot_sync_addrs, buffer_read_ptr, buffer_write_ptr, worker_noc_addr, message_size);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_write_worker_done_and_send_ack<write_to_worker>(
        buffer_slot_addrs,
        buffer_slot_sync_addrs,
        full_payload_size,
        buffer_read_ptr,
        buffer_write_ptr,
        num_messages_ack);
}

template <bool write_to_worker>
FORCE_INLINE void receiver_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& receiver_buffer_slot_sync_addrs,
    uint32_t message_size,
    uint32_t full_payload_size,
    uint32_t num_messages,
    uint64_t worker_noc_addr) {
    uint32_t total_msgs;
    if constexpr (measurement_type == MeasurementType::Latency) {
        total_msgs = num_messages;
    } else {
        total_msgs = num_messages * NUM_BUFFER_SLOTS;
    }

    DPRINT << "RECEIVER MAIN LOOP" << ENDL();

    uint32_t receiver_buffer_read_ptr = 0;
    uint32_t receiver_buffer_write_ptr = 0;
    uint32_t receiver_num_messages_ack = 0;

    if constexpr (write_to_worker) {
        noc_async_write_one_packet_with_trid_set_state(worker_noc_addr);
    }

    while (receiver_num_messages_ack < total_msgs) {
        update_receiver_state<write_to_worker>(
            receiver_buffer_slot_addrs,
            receiver_buffer_slot_sync_addrs,
            worker_noc_addr,
            message_size,
            full_payload_size,
            receiver_num_messages_ack,
            receiver_buffer_read_ptr,
            receiver_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}

// same as below so merge
template <bool write_to_worker>
FORCE_INLINE void send_receiver_bi_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& sender_buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& sender_buffer_slot_sync_addrs,
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
    const std::array<volatile eth_buffer_slot_sync_t*, NUM_BUFFER_SLOTS>& receiver_buffer_slot_sync_addrs,
    uint32_t full_payload_size,
    uint32_t message_size,
    uint32_t num_messages,
    uint64_t worker_noc_addr) {
    uint32_t total_msgs;
    if constexpr (measurement_type == MeasurementType::Latency) {
        total_msgs = num_messages * 2;
    } else {
        total_msgs = num_messages * NUM_BUFFER_SLOTS * 2;
    }

    DPRINT << "SENDER-RECEIVER MAIN LOOP" << ENDL();

    uint32_t sender_buffer_read_ptr = 0;
    uint32_t sender_buffer_write_ptr = 0;

    uint32_t receiver_buffer_read_ptr = 0;
    uint32_t receiver_buffer_write_ptr = 0;

    uint32_t num_messages_ack = 0;
    uint32_t sender_num_messages_send;
    if constexpr (measurement_type == MeasurementType::Latency) {
        sender_num_messages_send = num_messages;
    } else {
        sender_num_messages_send = num_messages * NUM_BUFFER_SLOTS;
    }

    if constexpr (write_to_worker) {
        noc_async_write_one_packet_with_trid_set_state(worker_noc_addr);
    }

    while (num_messages_ack < total_msgs) {
        update_sender_state(
            sender_buffer_slot_addrs,
            sender_buffer_slot_sync_addrs,
            full_payload_size,
            num_messages_ack,
            sender_num_messages_send,
            sender_buffer_read_ptr,
            sender_buffer_write_ptr);

        update_receiver_state<write_to_worker>(
            receiver_buffer_slot_addrs,
            receiver_buffer_slot_sync_addrs,
            worker_noc_addr,
            message_size,
            full_payload_size,
            num_messages_ack,
            receiver_buffer_read_ptr,
            receiver_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}
