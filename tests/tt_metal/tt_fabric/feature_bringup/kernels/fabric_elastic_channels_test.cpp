// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include <tuple>
#include "dataflow_api.h"
#include "kernels/fabric_elastic_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/hw/inc/compile_time_args.h"

// Compile-time configuration - passed as compile_args from host
constexpr size_t N_CHUNKS = get_compile_time_arg_val(0);
constexpr size_t RX_N_PKTS = get_compile_time_arg_val(1);
constexpr size_t CHUNK_N_PKTS = get_compile_time_arg_val(2);
constexpr size_t PACKET_SIZE = get_compile_time_arg_val(3);
constexpr bool BIDIRECTIONAL_MODE = get_compile_time_arg_val(4);

constexpr int32_t pkts_received_stream_id = 0;  // read by receiver, written by sender
constexpr int32_t pkts_acked_stream_id = 1;     // read by sender, written by receiver

using namespace tt::tt_fabric;

// Timing tracking structure
struct TimingStats {
    uint64_t total_acquire_cycles;
    uint64_t total_release_cycles;
    uint64_t total_test_cycles;
    uint32_t acquire_count;
    uint32_t release_count;

    TimingStats() :
        total_acquire_cycles(0), total_release_cycles(0), total_test_cycles(0), acquire_count(0), release_count(0) {}
};

// Global timing stats stored in L1 - will be initialized in kernel_main
volatile TimingStats* timing_stats = nullptr;

FORCE_INLINE int32_t get_receiver_num_unprocessed_packets() { return get_ptr_val(pkts_received_stream_id); }

FORCE_INLINE void receiver_mark_packet_processed() { increment_local_update_ptr_val(pkts_received_stream_id, -1); }

FORCE_INLINE void notify_remote_receiver_of_new_packet() { remote_update_ptr_val<pkts_received_stream_id, 0>(1); }

FORCE_INLINE int32_t get_sender_num_unprocessed_acks() { return get_ptr_val(pkts_acked_stream_id); }

FORCE_INLINE void sender_mark_ack_processed() { increment_local_update_ptr_val(pkts_acked_stream_id, -1); }

FORCE_INLINE void send_ack_to_sender() { remote_update_ptr_val<pkts_acked_stream_id, 0>(1); }

FORCE_INLINE void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        erisc_info->channels[0].bytes_sent = 0;
        erisc_info->channels[0].receiver_ack = 0;
        while (eth_txq_is_busy()) {
        }
        eth_send_bytes(handshake_register_address, handshake_register_address, 16);
        while (eth_txq_is_busy()) {
        }
        internal_::eth_send_packet(
            0,
            ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
            ((uint32_t)(&(erisc_info->channels[0].bytes_sent))) >> 4,
            1);
        while (erisc_info->channels[0].bytes_sent != 0) {
        }
    } else {
        while (erisc_info->channels[0].bytes_sent == 0) {
        }
        eth_receiver_channel_done(0);
    }
}

template <typename OpenChunksCbT, typename SendPoolT>
void process_send_side(
    volatile TimingStats* timing_stats,
    BufferIndex& current_chunk_slot_index,
    BufferIndex& current_chunk_ack_index,
    OpenChunksCbT& open_chunks_cb,
    SendPoolT& send_pool,
    size_t& unacked_sends,
    size_t& messages_sent,
    size_t& messages_acked,
    size_t messages_to_send,
    std::array<size_t, RX_N_PKTS>& receiver_buffer_addresses,
    ChannelBufferPointer<RX_N_PKTS>& sender_view_rx_buf_wr_ptr,
    size_t message_size) {
    // Process Ack
    auto num_unprocessed_acks = get_sender_num_unprocessed_acks();
    if (num_unprocessed_acks > 0) {
        current_chunk_ack_index = BufferIndex{static_cast<uint8_t>(current_chunk_ack_index.get() + 1)};
        bool can_release_chunk = current_chunk_ack_index.get() == CHUNK_N_PKTS;
        uint64_t start_cycles;
        uint64_t end_cycles;
        if (can_release_chunk) {  // 22 cycles!??!?! (44 -> 22 if I don't include the branch condition in the timing)

            start_cycles = eth_read_wall_clock();
            auto chunk = open_chunks_cb.pop();  // avg 7.3 cycles
            send_pool.return_chunk(chunk);      // 16 cycles
            current_chunk_ack_index = BufferIndex{0};
            end_cycles = eth_read_wall_clock();
        }
        sender_mark_ack_processed();
        if (can_release_chunk) {
            timing_stats->release_count++;
            timing_stats->total_release_cycles += (end_cycles - start_cycles);
        }
        messages_acked++;
        unacked_sends--;
    }

    // Try to send from current chunk
    if (!eth_txq_is_busy() && unacked_sends < RX_N_PKTS && messages_sent < messages_to_send &&
        !open_chunks_cb.is_empty() && current_chunk_slot_index < CHUNK_N_PKTS) {
        auto current_buffer = open_chunks_cb.peek_front();

        // Fill packet data
        auto src_addr = current_buffer->get_buffer_address(current_chunk_slot_index);
        auto dest_addr = receiver_buffer_addresses[sender_view_rx_buf_wr_ptr.get_buffer_index()];

        // Send packet using the buffer's channel ID
        eth_send_bytes_over_channel_payload_only_unsafe_one_packet(src_addr, dest_addr, message_size);

        while (eth_txq_is_busy()) {
        }
        notify_remote_receiver_of_new_packet();

        current_chunk_slot_index = BufferIndex{static_cast<uint8_t>(current_chunk_slot_index.get() + 1)};
        messages_sent++;

        sender_view_rx_buf_wr_ptr.increment();
        unacked_sends++;
    }

    // Acquire new chunk if current chunk is full
    if ((open_chunks_cb.is_empty() || current_chunk_slot_index == CHUNK_N_PKTS) && !send_pool.is_empty()) {
        current_chunk_slot_index = BufferIndex{0};
        // Acquire new chunk - measure timing
        uint64_t start_cycles = eth_read_wall_clock();
        auto new_chunk = send_pool.get_free_chunk();
        open_chunks_cb.push(new_chunk);
        uint64_t end_cycles = eth_read_wall_clock();

        timing_stats->total_acquire_cycles += (end_cycles - start_cycles);
        timing_stats->acquire_count++;
    }
}

void kernel_main() {
    init_ptr_val(pkts_received_stream_id, 0);
    init_ptr_val(pkts_acked_stream_id, 0);

    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t send_buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t timing_stats_addr = get_arg_val<uint32_t>(arg_idx++);

    // Initialize timing stats at address provided by host
    timing_stats = reinterpret_cast<volatile TimingStats*>(timing_stats_addr);
    timing_stats->total_acquire_cycles = 0;
    timing_stats->total_release_cycles = 0;
    timing_stats->acquire_count = 0;
    timing_stats->release_count = 0;

    // Initialize separate channel buffer pools for send and receive

    // Initialize pools dynamically based on N_CHUNKS
    std::array<std::tuple<size_t, uint8_t>, N_CHUNKS> send_buffers;
    std::array<size_t, RX_N_PKTS> recv_buffer_addresses;

    for (uint32_t i = 0; i < N_CHUNKS; i++) {
        send_buffers[i] = std::make_tuple(send_buffer_base + i * PACKET_SIZE * CHUNK_N_PKTS, static_cast<uint8_t>(i));
    }
    for (uint32_t i = 0; i < RX_N_PKTS; i++) {
        recv_buffer_addresses[i] = recv_buffer_base + i * PACKET_SIZE;
    }

    ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS> send_pool;
    send_pool.init(send_buffers, PACKET_SIZE, sizeof(eth_channel_sync_t));

    // Set up channels for bidirectional communication if enabled
    const uint32_t messages_per_direction = total_messages;

    // Setup handshake

    eth_setup_handshake(handshake_addr, is_sender_offset_0);

    size_t messages_acked = 0;
    size_t messages_sent = 0;
    size_t messages_received = 0;
    size_t total_messages_target = messages_per_direction;

    uint32_t idle_count = 0;
    const uint32_t idle_max = 100000;

    BufferIndex current_chunk_slot_index = BufferIndex{0};
    BufferIndex current_chunk_ack_index = BufferIndex{0};
#if defined(ARCH_WORMHOLE)
    // acq: 6.00
    // rel: 9.5
    WormholeEfficientCircularBuffer<typename ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS>::chunk_t*, N_CHUNKS>
        open_chunks_cb;
#else
    // acq: 17
    // rel: 4
    CircularBuffer<typename ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS>::chunk_t*, N_CHUNKS> open_chunks_cb;
#endif

    ChannelBufferPointer<RX_N_PKTS> sender_view_rx_buf_wr_ptr;
    size_t unacked_sends = 0;
    {
        DeviceZoneScopedN("FABRIC-ELASTIC-CHANNELS-TEST");

        auto test_start_cycles = eth_read_wall_clock();

        while (messages_acked < total_messages_target || messages_received < total_messages_target) {
            bool made_progress = false;

            if (BIDIRECTIONAL_MODE || is_sender_offset_0) {
                process_send_side(
                    timing_stats,
                    current_chunk_slot_index,
                    current_chunk_ack_index,
                    open_chunks_cb,
                    send_pool,
                    unacked_sends,
                    messages_sent,
                    messages_acked,
                    total_messages_target,

                    recv_buffer_addresses,
                    sender_view_rx_buf_wr_ptr,
                    message_size);
            }
            if (BIDIRECTIONAL_MODE || !is_sender_offset_0) {
                // Receiver logic - also use ChannelBuffersPool
                auto num_unprocessed_packets = get_receiver_num_unprocessed_packets();
                if (num_unprocessed_packets > 0 && !eth_txq_is_busy()) {
                    receiver_mark_packet_processed();
                    sender_view_rx_buf_wr_ptr.increment();
                    messages_received++;
                    send_ack_to_sender();
                }
            }

            // Idle management
            if (made_progress) {
                idle_count = 0;
            } else {
                idle_count++;
                if (idle_count > idle_max) {
                    run_routing();
                    idle_count = 0;
                }
            }
        }

        auto test_end_cycles = eth_read_wall_clock();
        timing_stats->total_test_cycles = test_end_cycles - test_start_cycles;
    }
}
