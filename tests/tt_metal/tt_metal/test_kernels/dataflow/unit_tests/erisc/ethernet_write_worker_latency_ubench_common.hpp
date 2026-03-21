// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <array>
#include <algorithm>
#include "eth_l1_address_map.h"
#include "api/dataflow/dataflow_api.h"
#include "internal/ethernet/dataflow_api.h"
#include "api/debug/assert.h"
#include "api/debug/dprint.h"
#include "eth_ubenchmark_types.hpp"
#include "risc_common.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

// #define ENABLE_DEBUG 1

// Overlay stream register IDs for flow control.
// Each core has these as LOCAL registers; the remote peer writes to them via eth_write_remote_reg.
// Stream 0: incremented remotely by peer when it sends a data packet to us
// Stream 1: incremented remotely by peer when it acks/completes processing of our sent packet
constexpr uint8_t STREAM_ID_PACKETS_FROM_REMOTE = 0;
constexpr uint8_t STREAM_ID_ACKS_FROM_REMOTE = 1;

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
constexpr bool is_power_of_two(T val) {
    return (val & (val - 1)) == T(0);
}

// ******************************* Common Ct Args ************************************************

constexpr BenchmarkType benchmark_type = static_cast<BenchmarkType>(get_compile_time_arg_val(0));
constexpr MeasurementType measurement_type = static_cast<MeasurementType>(get_compile_time_arg_val(1));
constexpr uint32_t NUM_BUFFER_SLOTS = get_compile_time_arg_val(2);
constexpr uint32_t MAX_NUM_TRANSACTION_ID = std::min<uint32_t>(
    NUM_BUFFER_SLOTS / 2, 8);  // the algorithm only works for NUM_BUFFER_SLOTS divisible by MAX_NUM_TRANSACTION_ID
constexpr uint32_t disable_trid = get_compile_time_arg_val(3);

// Initialize overlay stream registers for flow control.
// Must be called BEFORE handshake to ensure registers are ready before the remote peer starts.
FORCE_INLINE void init_flow_control_registers() {
    init_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>(0);
    init_ptr_val<STREAM_ID_ACKS_FROM_REMOTE>(0);
}

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

    return buffer_slot_addr;
}

template <uint32_t NUM_BUFFER_SLOTS>
FORCE_INLINE uint32_t advance_buffer_slot_ptr(uint32_t curr_ptr) {
    if constexpr (is_power_of_two(NUM_BUFFER_SLOTS)) {
        return (curr_ptr + 1) & (NUM_BUFFER_SLOTS - 1);
    } else if constexpr (NUM_BUFFER_SLOTS == 2) {
        return 1 - curr_ptr;
    } else {
        return curr_ptr == NUM_BUFFER_SLOTS - 1 ? 0 : curr_ptr + 1;
    }
}

FORCE_INLINE void update_sender_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t message_size,
    uint32_t& num_messages_ack,
    uint32_t& num_messages_send,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Try to send a packet if buffer has space and we have messages to send
    uint32_t next_write_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(buffer_write_ptr);
    bool buffer_not_full = next_write_ptr != buffer_read_ptr;

    if (buffer_not_full && num_messages_send != 0 && !eth_txq_is_busy()) {
        // Send data payload to remote (lands at same L1 address on peer)
        internal_::eth_send_packet_bytes_unsafe(
            0, buffer_slot_addrs[buffer_write_ptr], buffer_slot_addrs[buffer_write_ptr], message_size);
        // Wait for data send to complete, then notify receiver via stream register
        while (eth_txq_is_busy()) {
            switch_context_if_debug();
        }
        remote_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE, 0>(1);

        buffer_write_ptr = next_write_ptr;
        num_messages_send--;
    }

    // Check for acks from receiver via stream register
    int32_t new_acks = get_ptr_val<STREAM_ID_ACKS_FROM_REMOTE>();
    if (new_acks > 0) {
        increment_local_update_ptr_val<STREAM_ID_ACKS_FROM_REMOTE>(-new_acks);
        for (int32_t i = 0; i < new_acks; i++) {
            buffer_read_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(buffer_read_ptr);
        }
        num_messages_ack += new_acks;
    }
}

// ******************************* Receiver APIs *************************************************

FORCE_INLINE uint32_t setup_receiver_buffer(
    std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs, uint32_t buffer_slot_addr, uint32_t message_size) {
    for (uint8_t i = 0; i < NUM_BUFFER_SLOTS; i++) {
        buffer_slot_addrs[i] = buffer_slot_addr;
        buffer_slot_addr += message_size;
    }

    return buffer_slot_addr;
}

template <uint32_t MAX_NUM_TRANSACTION_ID>
FORCE_INLINE uint32_t get_buffer_slot_trid(uint32_t curr_ptr) {
    if constexpr (is_power_of_two(MAX_NUM_TRANSACTION_ID)) {
        return (curr_ptr + 1) & (MAX_NUM_TRANSACTION_ID - 1);
    } else if constexpr (MAX_NUM_TRANSACTION_ID == 2) {
        return 1 - curr_ptr;
    } else {
        return curr_ptr == MAX_NUM_TRANSACTION_ID - 1 ? 0 : curr_ptr + 1;
    }
}

FORCE_INLINE bool write_worker_done(uint32_t trid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_sent(noc_index, trid);
}

FORCE_INLINE void write_worker(
    uint32_t buffer_slot_addr, uint64_t worker_noc_addr, uint32_t message_size, uint32_t curr_trid_to_write) {
#ifdef DISABLE_TRID
    noc_async_write(buffer_slot_addr, worker_noc_addr, message_size);
    noc_async_writes_flushed();
#else
    noc_async_write_one_packet_with_trid_with_state<false, false>(
        buffer_slot_addr, worker_noc_addr, message_size, curr_trid_to_write);
#endif
}

template <bool write_to_worker, uint32_t txq_id = 0>
FORCE_INLINE void update_receiver_state(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t& num_messages_ack,
    uint32_t& buffer_read_ptr,
    uint32_t& buffer_write_ptr) {
    // Check for incoming packets via stream register
    uint32_t next_write_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(buffer_write_ptr);
    bool buffer_not_full = next_write_ptr != buffer_read_ptr;

    if (buffer_not_full && get_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>() > 0) {
        increment_local_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>(-1);
        if constexpr (write_to_worker) {
            uint32_t curr_trid = get_buffer_slot_trid<MAX_NUM_TRANSACTION_ID>(buffer_write_ptr);
            write_worker(buffer_slot_addrs[buffer_write_ptr], worker_noc_addr, message_size, curr_trid);
        }
        buffer_write_ptr = next_write_ptr;
    }

    // Send ack when processing is complete
    bool buffer_not_empty = buffer_read_ptr != buffer_write_ptr;
    bool send_ack_condition = buffer_not_empty;
    if constexpr (write_to_worker and !disable_trid) {
        uint32_t curr_trid = get_buffer_slot_trid<MAX_NUM_TRANSACTION_ID>(buffer_read_ptr);
        send_ack_condition = send_ack_condition && write_worker_done(curr_trid);
    }
    if (send_ack_condition) {
        if constexpr (measurement_type == MeasurementType::Latency) {
            // Send data back for symmetric round-trip measurement
            while (internal_::eth_txq_is_busy(txq_id)) {
                switch_context_if_debug();
            }
            internal_::eth_send_packet_bytes_unsafe(
                txq_id, buffer_slot_addrs[buffer_read_ptr], buffer_slot_addrs[buffer_read_ptr], message_size);
        }
        // Send ack via stream register
        while (internal_::eth_txq_is_busy(txq_id)) {
            switch_context_if_debug();
        }
        remote_update_ptr_val<STREAM_ID_ACKS_FROM_REMOTE, txq_id>(1);

        buffer_read_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(buffer_read_ptr);
        num_messages_ack++;
    }
}

template <bool write_to_worker, uint32_t txq_id = 0>
FORCE_INLINE void receiver_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
    uint32_t message_size,
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
        update_receiver_state<write_to_worker, txq_id>(
            receiver_buffer_slot_addrs,
            worker_noc_addr,
            message_size,
            receiver_num_messages_ack,
            receiver_buffer_read_ptr,
            receiver_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}

// ******************************* Dual-ERISC Sender/Receiver APIs *********************************
// For DualEriscBiDir mode: ERISC_0 (sender) uses TXQ0 for data + notifications (stream reg writes).
// ERISC_1 (receiver) uses TXQ1 for acks as DATA packets (register writes via TXQ1 don't work).
// Sender polls an L1 ack counter instead of stream registers for acks.

// Sender loop for DualEriscBiDir: uses L1 ack counter instead of stream register acks
FORCE_INLINE void dual_erisc_send_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& buffer_slot_addrs,
    uint32_t message_size,
    uint32_t num_messages,
    volatile tt_l1_ptr uint32_t* ack_counter) {
    uint32_t total_msgs;
    if constexpr (measurement_type == MeasurementType::Latency) {
        total_msgs = num_messages;
    } else {
        total_msgs = num_messages * NUM_BUFFER_SLOTS;
    }

    uint32_t sender_buffer_read_ptr = 0;
    uint32_t sender_buffer_write_ptr = 0;
    uint32_t sender_num_messages_ack = 0;
    uint32_t sender_num_messages_send = total_msgs;
    uint32_t last_seen_ack = 0;

    while (sender_num_messages_ack < total_msgs) {
        // Try to send a packet if buffer has space and we have messages to send
        uint32_t next_write_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(sender_buffer_write_ptr);
        bool buffer_not_full = next_write_ptr != sender_buffer_read_ptr;

        if (buffer_not_full && sender_num_messages_send != 0 && !eth_txq_is_busy()) {
            internal_::eth_send_packet_bytes_unsafe(
                0,
                buffer_slot_addrs[sender_buffer_write_ptr],
                buffer_slot_addrs[sender_buffer_write_ptr],
                message_size);
            while (eth_txq_is_busy()) {
                switch_context_if_debug();
            }
            remote_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE, 0>(1);
            sender_buffer_write_ptr = next_write_ptr;
            sender_num_messages_send--;
        }

        // Check for acks via L1 counter (written by remote ERISC_1 via TXQ1 data packet)
        invalidate_l1_cache();
        uint32_t current_ack = *ack_counter;
        if (current_ack > last_seen_ack) {
            uint32_t new_acks = current_ack - last_seen_ack;
            last_seen_ack = current_ack;
            for (uint32_t i = 0; i < new_acks; i++) {
                sender_buffer_read_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(sender_buffer_read_ptr);
            }
            sender_num_messages_ack += new_acks;
        }

        switch_context_if_debug();
    }
}

// Receiver loop for DualEriscBiDir: sends acks as data packets via TXQ1
// ack_counter_addr: the REMOTE sender's L1 address where it polls for acks (destination of TXQ1 data send)
// outgoing_ack_addr: LOCAL L1 address used as source for TXQ1 data send (must not conflict with sender's ack poll)
template <bool write_to_worker>
FORCE_INLINE void dual_erisc_receiver_uni_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
    uint32_t message_size,
    uint32_t num_messages,
    uint64_t worker_noc_addr,
    uint32_t remote_ack_dest_addr,
    uint32_t local_ack_src_addr) {
    uint32_t total_msgs;
    if constexpr (measurement_type == MeasurementType::Latency) {
        total_msgs = num_messages;
    } else {
        total_msgs = num_messages * NUM_BUFFER_SLOTS;
    }

    uint32_t receiver_buffer_read_ptr = 0;
    uint32_t receiver_buffer_write_ptr = 0;
    uint32_t receiver_num_messages_ack = 0;

    // Local ack counter: incremented each time we ack, then sent to remote sender via TXQ1.
    // Uses a SEPARATE L1 address from the incoming ack counter to avoid conflicting with
    // the local sender (ERISC_0) which polls ack_counter_addr for ITS incoming acks.
    volatile tt_l1_ptr uint32_t* local_ack = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(local_ack_src_addr);
    *local_ack = 0;

    if constexpr (write_to_worker) {
        noc_async_write_one_packet_with_trid_set_state(worker_noc_addr);
    }

    while (receiver_num_messages_ack < total_msgs) {
        // Check for incoming packets via stream register
        uint32_t next_write_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(receiver_buffer_write_ptr);
        bool buffer_not_full = next_write_ptr != receiver_buffer_read_ptr;

        if (buffer_not_full && get_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>() > 0) {
            increment_local_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>(-1);
            if constexpr (write_to_worker) {
                uint32_t curr_trid = get_buffer_slot_trid<MAX_NUM_TRANSACTION_ID>(receiver_buffer_write_ptr);
                write_worker(
                    receiver_buffer_slot_addrs[receiver_buffer_write_ptr], worker_noc_addr, message_size, curr_trid);
            }
            receiver_buffer_write_ptr = next_write_ptr;
        }

        // Send ack when processing is complete
        bool buffer_not_empty = receiver_buffer_read_ptr != receiver_buffer_write_ptr;
        bool send_ack_condition = buffer_not_empty;
        if constexpr (write_to_worker and !disable_trid) {
            uint32_t curr_trid = get_buffer_slot_trid<MAX_NUM_TRANSACTION_ID>(receiver_buffer_read_ptr);
            send_ack_condition = send_ack_condition && write_worker_done(curr_trid);
        }
        if (send_ack_condition) {
            if constexpr (measurement_type == MeasurementType::Latency) {
                // Send data back for symmetric round-trip measurement
                while (internal_::eth_txq_is_busy(1)) {
                    switch_context_if_debug();
                }
                internal_::eth_send_packet_bytes_unsafe(
                    1,
                    receiver_buffer_slot_addrs[receiver_buffer_read_ptr],
                    receiver_buffer_slot_addrs[receiver_buffer_read_ptr],
                    message_size);
            }
            // Increment local ack counter and send to remote sender via TXQ1 data packet.
            // IMPORTANT: Wait for TXQ1 to finish previous DMA BEFORE writing new ack value to L1,
            // otherwise we corrupt in-flight DMA source data.
            receiver_num_messages_ack++;
            while (internal_::eth_txq_is_busy(1)) {
                switch_context_if_debug();
            }
            *local_ack = receiver_num_messages_ack;
            // Send from local_ack_src_addr to remote's ack_counter_addr (incoming ack location)
            internal_::eth_send_packet_bytes_unsafe(1, local_ack_src_addr, remote_ack_dest_addr, 16);

            receiver_buffer_read_ptr = advance_buffer_slot_ptr<NUM_BUFFER_SLOTS>(receiver_buffer_read_ptr);
        }

        switch_context_if_debug();
    }
}

// ******************************* Tensix-Eth-Eth-Tensix Sender/Receiver APIs ***********************

// Sender eth loop for TensixEthEthTensixUniDir benchmark.
// Sender eth spins on push_counter from local tensix, sends over ethernet, waits for ack,
// then writes returning echo data to local tensix and signals tensix_sem.
FORCE_INLINE void tensix_eth_sender_loop(
    uint32_t data_slot_addr,
    uint32_t push_counter_addr,
    uint64_t tensix_landing_noc_addr,
    uint64_t tensix_sem_noc_addr,
    uint32_t message_size,
    uint32_t num_messages) {
    volatile tt_l1_ptr uint32_t* push_counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(push_counter_addr);

    // Signal tensix that eth is ready (handshake complete, push_counter initialized)
    noc_semaphore_inc(tensix_sem_noc_addr, 1);

    for (uint32_t i = 0; i < num_messages; i++) {
        // Wait for local tensix to push data into our slot
        invalidate_l1_cache();
        while (*push_counter == 0) {
            invalidate_l1_cache();
        }
        *push_counter = 0;

        // Send data over ethernet to remote eth peer
        internal_::eth_send_packet_bytes_unsafe(0, data_slot_addr, data_slot_addr, message_size);
        while (eth_txq_is_busy()) {
            switch_context_if_debug();
        }
        // Notify remote eth that a packet arrived
        remote_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE, 0>(1);

        // Wait for ack (echo data returned from remote)
        while (get_ptr_val<STREAM_ID_ACKS_FROM_REMOTE>() == 0) {
            switch_context_if_debug();
        }
        increment_local_update_ptr_val<STREAM_ID_ACKS_FROM_REMOTE>(-1);

        // Write echo data from our slot to local tensix landing buffer
        noc_async_write(data_slot_addr, tensix_landing_noc_addr, message_size);

        // Signal local tensix that data is ready (NOC ordering guarantees data arrives first)
        noc_semaphore_inc(tensix_sem_noc_addr, 1);
    }
}

// Receiver eth loop for TensixEthEthTensixUniDir benchmark.
// Receiver eth spins on overlay reg for incoming packet, writes to local tensix, signals tensix_sem,
// then waits for tensix to echo data back, sends echo over ethernet, and acks sender.
FORCE_INLINE void tensix_eth_receiver_loop(
    uint32_t data_slot_addr,
    uint32_t push_counter_addr,
    uint64_t tensix_landing_noc_addr,
    uint64_t tensix_sem_noc_addr,
    uint32_t message_size,
    uint32_t num_messages) {
    volatile tt_l1_ptr uint32_t* push_counter = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(push_counter_addr);

    // Signal tensix that eth is ready (handshake complete, push_counter initialized)
    noc_semaphore_inc(tensix_sem_noc_addr, 1);

    for (uint32_t i = 0; i < num_messages; i++) {
        // Wait for incoming packet from remote eth (sender)
        while (get_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>() == 0) {
            switch_context_if_debug();
        }
        increment_local_update_ptr_val<STREAM_ID_PACKETS_FROM_REMOTE>(-1);

        // Write received data to local tensix landing buffer
        noc_async_write(data_slot_addr, tensix_landing_noc_addr, message_size);

        // Signal local tensix that data is ready (NOC ordering guarantees data arrives first)
        noc_semaphore_inc(tensix_sem_noc_addr, 1);

        // Wait for local tensix to push echo data back into our slot
        invalidate_l1_cache();
        while (*push_counter == 0) {
            invalidate_l1_cache();
        }
        *push_counter = 0;

        // Send echo data back over ethernet to sender eth
        internal_::eth_send_packet_bytes_unsafe(0, data_slot_addr, data_slot_addr, message_size);
        while (eth_txq_is_busy()) {
            switch_context_if_debug();
        }
        // Ack the sender eth
        remote_update_ptr_val<STREAM_ID_ACKS_FROM_REMOTE, 0>(1);
    }
}

// same as below so merge
template <bool write_to_worker>
FORCE_INLINE void send_receiver_bi_dir(
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& sender_buffer_slot_addrs,
    const std::array<uint32_t, NUM_BUFFER_SLOTS>& receiver_buffer_slot_addrs,
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
            message_size,
            num_messages_ack,
            sender_num_messages_send,
            sender_buffer_read_ptr,
            sender_buffer_write_ptr);

        update_receiver_state<write_to_worker>(
            receiver_buffer_slot_addrs,
            worker_noc_addr,
            message_size,
            num_messages_ack,
            receiver_buffer_read_ptr,
            receiver_buffer_write_ptr);

        // not called in normal execution mode
        switch_context_if_debug();
    }
}
