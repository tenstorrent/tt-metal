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

// static constexpr uint32_t MAX_TRANSACTION_ID = 15;
// static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);
// static constexpr uint32_t worker_noc_x = get_compile_time_arg_val(1);
// static constexpr uint32_t worker_noc_y = get_compile_time_arg_val(2);
// static constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(3);
// static constexpr bool use_transaction_id = get_compile_time_arg_val(4) == 1;
// static constexpr uint32_t num_writes_skip_barrier = get_compile_time_arg_val(5);

// template <bool sending_tails>
// FORCE_INLINE void run_loop_iteration(
//     const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
//     const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
//     uint64_t worker_noc_addr,
//     uint32_t message_size) {
//     if constexpr (!sending_tails) {
//         static uint32_t writes_count;

//         for (uint32_t i = 0; i < NUM_CHANNELS; i++) {
//             if constexpr (use_transaction_id) {
//                 uint32_t curr_transaction_id = i % MAX_TRANSACTION_ID + 1;

//                 if (writes_count < num_writes_skip_barrier) {
//                     // wait for sender data arrive
//                     while (channel_sync_addrs[i]->bytes_sent == 0) {
//                     }

//                     noc_async_write_with_trid(channel_addrs[i], worker_noc_addr, message_size, curr_transaction_id);

//                     channel_sync_addrs[i]->bytes_sent = 0;

//                     DPRINT << "write CH: " << i << " tid: " << curr_transaction_id << ENDL();

//                     // not using any barrier
//                     writes_count++;
//                 } else {
//                     uint32_t prev_channel_id_to_ack = (NUM_CHANNELS - num_writes_skip_barrier + i) % NUM_CHANNELS;
//                     uint32_t prev_transaction_id = prev_channel_id_to_ack % MAX_TRANSACTION_ID + 1;

//                     // barrier on previous data
//                     noc_async_write_barrier_with_trid(prev_transaction_id);

//                     channel_sync_addrs[prev_channel_id_to_ack]->bytes_sent = 0;

//                     // wait for txq to be ready, otherwise we'll
//                     // hit a context switch in the send command
//                     while (eth_txq_is_busy()) {
//                     }
//                     eth_send_bytes_over_channel_payload_only_unsafe(
//                         reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
//                         reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
//                         sizeof(eth_channel_sync_t),
//                         sizeof(eth_channel_sync_t),
//                         sizeof(eth_channel_sync_t) >> 4);

//                     DPRINT << "ack CH: " << prev_channel_id_to_ack << ENDL();
//                     DPRINT << "wait tid: " << prev_transaction_id << ENDL();

//                     // wait only after the ack is sent for the previous packet
//                     while (channel_sync_addrs[i]->bytes_sent == 0) {
//                     }

//                     noc_async_write_with_trid(channel_addrs[i], worker_noc_addr, message_size, curr_transaction_id);

//                     channel_sync_addrs[i]->bytes_sent = 0;

//                     DPRINT << "write CH: " << i << " tid: " << curr_transaction_id << ENDL();
//                 }
//             } else {
//                 // wait for sender data arrive
//                 while (channel_sync_addrs[i]->bytes_sent == 0) {
//                 }

//                 channel_sync_addrs[i]->bytes_sent = 0;

//                 // for current channel data, send it to worker core and perform barrier
//                 noc_async_write(channel_addrs[i], worker_noc_addr, message_size);
//                 noc_async_write_barrier();

//                 // wait for txq to be ready, otherwise we'll
//                 // hit a context switch in the send command
//                 while (eth_txq_is_busy()) {
//                 }
//                 eth_send_bytes_over_channel_payload_only_unsafe(
//                     reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
//                     reinterpret_cast<uint32_t>(channel_sync_addrs[i]),
//                     sizeof(eth_channel_sync_t),
//                     sizeof(eth_channel_sync_t),
//                     sizeof(eth_channel_sync_t) >> 4);
//             }
//         }
//     } else {
//         // sending tails
//         for (uint32_t i = 0; i < num_writes_skip_barrier; i++) {
//             // barrier on previous data
//             uint32_t prev_channel_id_to_ack = NUM_CHANNELS - num_writes_skip_barrier + i;
//             uint32_t prev_transaction_id = prev_channel_id_to_ack % MAX_TRANSACTION_ID + 1;

//             noc_async_write_barrier_with_trid(prev_transaction_id);

//             DPRINT << "tails wait tid: " << prev_transaction_id << ENDL();
//             DPRINT << "tails ack CH: " << prev_channel_id_to_ack << ENDL();

//             channel_sync_addrs[prev_channel_id_to_ack]->bytes_sent = 0;

//             // wait for txq to be ready, otherwise we'll
//             // hit a context switch in the send command
//             while (eth_txq_is_busy()) {
//             }
//             eth_send_bytes_over_channel_payload_only_unsafe(
//                 reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
//                 reinterpret_cast<uint32_t>(channel_sync_addrs[prev_channel_id_to_ack]),
//                 sizeof(eth_channel_sync_t),
//                 sizeof(eth_channel_sync_t),
//                 sizeof(eth_channel_sync_t) >> 4);
//         }
//     }
// }

static constexpr uint32_t NUM_CHANNELS = get_compile_time_arg_val(0);
static constexpr uint32_t MAX_NUM_TRANSACTION_ID =
    NUM_CHANNELS / 2;  // the algorithm only works for NUM_CHANNELS divisible by MAX_NUM_TRANSACTION_ID
static constexpr uint32_t worker_noc_x = get_compile_time_arg_val(1);
static constexpr uint32_t worker_noc_y = get_compile_time_arg_val(2);
static constexpr uint32_t worker_buffer_addr = get_compile_time_arg_val(3);
static constexpr bool use_transaction_id = get_compile_time_arg_val(4) == 1;
static constexpr uint32_t num_writes_skip_barrier = get_compile_time_arg_val(5);

FORCE_INLINE bool has_incoming_packet(volatile eth_channel_sync_t* channel_sync_addr) {
    return channel_sync_addr->bytes_sent != 0;
}

FORCE_INLINE bool write_done(uint32_t tid) {
    return ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, tid);
}

FORCE_INLINE void ack_complete(volatile eth_channel_sync_t* channel_sync_addr) {
    channel_sync_addr->bytes_sent = 0;

    // wait for txq to be ready, otherwise we'll
    // hit a context switch in the send command
    while (eth_txq_is_busy()) {
#if ENABLE_DEBUG
        internal_::risc_context_switch();
#endif
    }
#if ENABLE_DEBUG
    eth_send_bytes_over_channel_payload_only(
#else
    eth_send_bytes_over_channel_payload_only_unsafe(
#endif
        reinterpret_cast<uint32_t>(channel_sync_addr),
        reinterpret_cast<uint32_t>(channel_sync_addr),
        sizeof(eth_channel_sync_t),
        sizeof(eth_channel_sync_t),
        sizeof(eth_channel_sync_t) >> 4);
}

FORCE_INLINE void write_to_worker(
    uint32_t channel_addr,
    volatile eth_channel_sync_t* channel_sync_addr,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t curr_tid_to_write) {
    // write to local
    noc_async_write_one_packet_with_trid(channel_addr, worker_noc_addr, message_size, curr_tid_to_write);

    // reset sync
    channel_sync_addr->bytes_sent = 0;
}

FORCE_INLINE void process_messages(
    const std::array<uint32_t, NUM_CHANNELS>& channel_addrs,
    const std::array<volatile eth_channel_sync_t*, NUM_CHANNELS>& channel_sync_addrs,
    uint64_t worker_noc_addr,
    uint32_t message_size,
    uint32_t num_messages) {
    uint32_t total_msgs = num_messages * NUM_CHANNELS;

    uint32_t chCount = 0;
    uint32_t tidCount = 0;

    DPRINT << "MAIN LOOP" << ENDL();

    // Variables to hold the pointer values
    uint32_t ch = 0;
    uint32_t tid = 0;
    uint32_t curr_ch_to_ack = 0;
    uint32_t curr_tid_to_write = 0;

    uint32_t i = 0;
    while (i < total_msgs) {
        ch = chCount % NUM_CHANNELS;                    // range: 0..17
        tid = (tidCount % MAX_NUM_TRANSACTION_ID) + 1;  // range: 1..9

        // 1) Check if there's an incoming packet for ch
        if (has_incoming_packet(channel_sync_addrs[ch])) {
            curr_tid_to_write = ch % MAX_NUM_TRANSACTION_ID + 1;
            write_to_worker(
                channel_addrs[ch], channel_sync_addrs[ch], worker_noc_addr, message_size, curr_tid_to_write);

            DPRINT << "write from ch: " << chCount << " with tid: " << curr_tid_to_write << ENDL();

            // Only increment chCount if we won't exceed tidCount + 17
            // i.e. chCount < tidCount + 18
            if (chCount < tidCount + NUM_CHANNELS) {
                chCount++;
            }

            i++;
        }

        // 2) Check if the write for tid is done, make sure tid count is less than ch count so we never check barrier on
        // the packet hasn't been sent
        if (write_done(tid) && (tidCount < chCount)) {
            curr_ch_to_ack = tidCount % NUM_CHANNELS;
            ack_complete(channel_sync_addrs[curr_ch_to_ack]);

            DPRINT << "ack to ch: " << curr_ch_to_ack << ENDL();

            tidCount++;
        }

#if ENABLE_DEBUG
        internal_::risc_context_switch();
#endif
    }

    DPRINT << "ACK TAIL" << ENDL();

    // wait for some tail writes to finish so we can send ack to sender
    while (tidCount < chCount) {
        tid = (tidCount % MAX_NUM_TRANSACTION_ID) + 1;
        if (write_done(tid)) {
            curr_ch_to_ack = tidCount % NUM_CHANNELS;
            ack_complete(channel_sync_addrs[curr_ch_to_ack]);

            DPRINT << "ack to ch: " << curr_ch_to_ack << ENDL();

            tidCount++;
        }

#if ENABLE_DEBUG
        internal_::risc_context_switch();
#endif
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
        process_messages(channel_addrs, channel_sync_addrs, worker_noc_addr, message_size, num_messages);
        // for (uint32_t i = 0; i < num_messages; i++) {
        //     run_loop_iteration<false>(channel_addrs, channel_sync_addrs, worker_noc_addr, message_size);
        // }

        // if constexpr (use_transaction_id) {
        //     run_loop_iteration<true>(channel_addrs, channel_sync_addrs, worker_noc_addr, message_size);
        // }
    }
}
