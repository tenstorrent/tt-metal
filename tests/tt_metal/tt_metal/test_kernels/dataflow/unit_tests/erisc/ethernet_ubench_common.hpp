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

// #define ENABLE_DEBUG 1

/*
1. sender sends packet if local view of space in receiver buffer shows that there is space
2. sender decrements local view of space in receiver buffer
3. sender increments register to signal that it send packet
4. receiver reads its local view of whether packet has been sent
     (can just check for non-zero)
5. if there is a packet then receiver decrements its local view of the register
     that sender incremented in 3 and processes packet
6. receiver sends ack to sender by incrementing view of space in receiver buffer
*/

// Assign streams for sender <-> receiver flow control ptrs
// Update this stream when senders send packet to receiver
constexpr uint32_t to_receiver_pkts_sent_id = 0;
// Receivers updates the reg on this stream to signal to sender it can receive
constexpr uint32_t receiver_buffer_availability_id = 1;

template <uint32_t stream_id>
FORCE_INLINE void init_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG(stream_id, STREAM_REMOTE_DEST_BUF_SIZE_REG_INDEX, val);
}

// This will be an atomic register read to the register
template <uint32_t stream_id>
FORCE_INLINE int32_t get_ptr_val() {
    return NOC_STREAM_READ_REG(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_REG_INDEX);
}

// Writing to this register will leverage the built-in stream hardware which will automatically perform an atomic
// increment on the register. This can save precious erisc cycles by offloading a lot of pointer manipulation.
// Additionally, these registers are accessible via eth_reg_write calls which can be used to write a value,
// inline the eth command (without requiring source L1)
template <uint32_t stream_id>
FORCE_INLINE void increment_local_update_ptr_val(int32_t val) {
    NOC_STREAM_WRITE_REG_FIELD(
        stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX, REMOTE_DEST_BUF_WORDS_FREE_INC, val);
}

template <uint32_t stream_id>
FORCE_INLINE void remote_update_ptr_val(int32_t val) {
    constexpr uint32_t addr = STREAM_REG_ADDR(stream_id, STREAM_REMOTE_DEST_BUF_SPACE_AVAILABLE_UPDATE_REG_INDEX);
    internal_::eth_write_remote_reg_no_txq_check(0, addr, val << REMOTE_DEST_BUF_WORDS_FREE_INC);
}

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
constexpr uint32_t worker_noc_x = get_compile_time_arg_val(3);
constexpr uint32_t worker_noc_y = get_compile_time_arg_val(4);
constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(5);
constexpr uint32_t disable_trid = get_compile_time_arg_val(6);

// ******************************* Sender APIs ***************************************************

FORCE_INLINE uint32_t setup_sender_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs, uint32_t buffer_slot_addr, uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
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
    uint32_t buffer_slot_addr, uint32_t full_payload_size, int32_t& sender_view_of_receiver_buffer) {
    increment_local_update_ptr_val<receiver_buffer_availability_id>(-1);
    sender_view_of_receiver_buffer = get_ptr_val<receiver_buffer_availability_id>();

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }

    eth_send_bytes_over_channel_payload_only_unsafe_one_packet(buffer_slot_addr, buffer_slot_addr, full_payload_size);

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }

    remote_update_ptr_val<to_receiver_pkts_sent_id>(1);
}

FORCE_INLINE bool has_receiver_ack(int32_t sender_view_of_receiver_buffer) {
    return get_ptr_val<receiver_buffer_availability_id>() > sender_view_of_receiver_buffer;
}

FORCE_INLINE void check_buffer_full_and_send_packet(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t full_payload_size,
    uint32_t& num_messages_send,
    int32_t& sender_view_of_receiver_buffer) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    // bool buffer_not_full = next_write_ptr != read_ptr;
    bool buffer_not_full = (get_ptr_val<receiver_buffer_availability_id>() != 0);

    if (buffer_not_full && num_messages_send != 0) {
        write_receiver(buffer_slot_addrs[write_ptr], full_payload_size, sender_view_of_receiver_buffer);

        write_ptr = next_write_ptr;
        num_messages_send--;
    }
}

FORCE_INLINE void check_receiver_done(
    uint32_t& read_ptr, uint32_t& num_messages_ack, int32_t sender_view_of_receiver_buffer) {
    if (has_receiver_ack(sender_view_of_receiver_buffer)) {
        uint32_t next_read_ptr = advance_buffer_slot_ptr(read_ptr);

        read_ptr = next_read_ptr;
        num_messages_ack++;
    }
}

FORCE_INLINE void update_sender_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t full_payload_size,
    uint32_t& num_messages_ack,
    uint32_t& num_messages_send,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr,
    int32_t& sender_view_of_receiver_buffer) {
    // Check if current buffer slot is ready and send packet to receiver
    check_buffer_full_and_send_packet(
        buffer_slot_addrs,
        buffer_read_ptr,
        buffer_write_ptr,
        full_payload_size,
        num_messages_send,
        sender_view_of_receiver_buffer);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_receiver_done(buffer_read_ptr, num_messages_ack, sender_view_of_receiver_buffer);
}

// ******************************* Receiver APIs *************************************************

FORCE_INLINE uint32_t setup_receiver_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs, uint32_t buffer_slot_addr, uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
    }

    uint32_t buffer_end_addr = buffer_slot_addr;
    return buffer_end_addr;
}

FORCE_INLINE uint32_t get_buffer_slot_trid(uint32_t curr_ptr) { return curr_ptr % MAX_NUM_TRANSACTION_ID + 1; }

FORCE_INLINE bool has_incoming_packet() { return get_ptr_val<to_receiver_pkts_sent_id>() != 0; }

FORCE_INLINE bool write_worker_done(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, trid);
}

FORCE_INLINE void ack_complete(uint32_t buffer_slot_addr, uint32_t full_payload_size) {
    increment_local_update_ptr_val<to_receiver_pkts_sent_id>(-1);

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }

    if constexpr (measurement_type == MeasurementType::Latency) {
        // Send pack entire packet so measurement from sender -> receiver -> sender is symmetric
        eth_send_bytes_over_channel_payload_only_unsafe_one_packet(
            buffer_slot_addr, buffer_slot_addr, full_payload_size);
    }

    while (eth_txq_is_busy()) {
        switch_context_if_debug();
    }
    remote_update_ptr_val<receiver_buffer_availability_id>(1);
}

FORCE_INLINE void write_worker(
    uint32_t buffer_slot_addr, uint64_t worker_noc_addr, uint32_t message_size, uint32_t curr_trid_to_write) {
    // write to local
    if constexpr (disable_trid) {
        noc_async_write(buffer_slot_addr, worker_noc_addr, message_size);
        noc_async_writes_flushed();
    } else {
        noc_async_write_one_packet_with_trid_with_state(
            buffer_slot_addr, worker_noc_addr, message_size, curr_trid_to_write);
    }
}

template <bool write_to_worker>
FORCE_INLINE void check_incoming_packet_and_write_worker(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t read_ptr,
    uint32_t& write_ptr,
    uint64_t worker_noc_addr,
    uint32_t message_size) {
    uint32_t next_write_ptr = advance_buffer_slot_ptr(write_ptr);
    bool buffer_not_full = next_write_ptr != read_ptr;

    if (buffer_not_full && has_incoming_packet()) {
        if constexpr (write_to_worker) {
            int count = 0;
            while (count < 100) {
                count++;
            }

            uint32_t curr_trid = get_buffer_slot_trid(write_ptr);
            write_worker(buffer_slot_addrs[write_ptr], worker_noc_addr, message_size, curr_trid);
        }
        write_ptr = next_write_ptr;
    }
}

template <bool write_to_worker>
FORCE_INLINE void check_write_worker_done_and_send_ack(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
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
        ack_complete(buffer_slot_addrs[read_ptr], full_payload_size);
        read_ptr = advance_buffer_slot_ptr(read_ptr);
        num_messages_ack++;
    }
}

template <bool write_to_worker>
FORCE_INLINE void update_receiver_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t full_payload_size,
    uint32_t& num_messages_ack,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check if there's an incoming packet for current buffer slot and write to worker if there's new packet
    check_incoming_packet_and_write_worker<write_to_worker>(
        buffer_slot_addrs, buffer_read_ptr, buffer_write_ptr, worker_noc_addr, message_size);
    // Check if the write for trid is done, and ack sender if the current buffer slot is done
    check_write_worker_done_and_send_ack<write_to_worker>(
        buffer_slot_addrs, full_payload_size, buffer_read_ptr, buffer_write_ptr, num_messages_ack);
}

template <bool write_to_worker>
FORCE_INLINE void receiver_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
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
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
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
    int32_t sender_view_of_receiver_buffer = get_ptr_val<receiver_buffer_availability_id>();

    if constexpr (write_to_worker) {
        noc_async_write_one_packet_with_trid_set_state(worker_noc_addr);
    }

    while (num_messages_ack < total_msgs) {
        update_sender_state(
            sender_buffer_slot_addrs,
            full_payload_size,
            num_messages_ack,
            sender_num_messages_send,
            sender_buffer_read_ptr,
            sender_buffer_write_ptr,
            sender_view_of_receiver_buffer);

        update_receiver_state<write_to_worker>(
            receiver_buffer_slot_addrs,
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
