// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include <tuple>
#include "dataflow_api.h"
#include "fabric_elastic_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp"
#include "tt_metal/hw/inc/compile_time_args.h"
#include "core_config.h"

// Compile-time configuration - passed as compile_args from host
constexpr size_t N_CHUNKS = get_compile_time_arg_val(0);
constexpr size_t RX_N_PKTS = get_compile_time_arg_val(1);
constexpr size_t CHUNK_N_PKTS = get_compile_time_arg_val(2);
constexpr size_t PACKET_SIZE_BYTES = get_compile_time_arg_val(3);
constexpr bool BIDIRECTIONAL_MODE = get_compile_time_arg_val(4);
constexpr size_t N_SRC_CHANS = get_compile_time_arg_val(5);

// pkts received (from worker) stream IDs are [0, N_SRC_CHANS-1]
// pkts acked (to sender from receiver) stream IDs are [N_SRC_CHANS, 2*N_SRC_CHANS-1]

constexpr int32_t pkts_received_stream_id = 2 * N_SRC_CHANS;  // read by receiver, written by sender

static_assert(pkts_received_stream_id < 10, "pkts_received_stream_id must be less than 10");
static_assert(PACKET_SIZE_BYTES % sizeof(uint32_t) == 0, "PACKET_SIZE_BYTES must be a multiple of sizeof(uint32_t)");

using tt::tt_fabric::BufferIndex;

// Points to a chunk and steps through the addresses
using chunk_forward_iterator_t =
    tt::tt_fabric::OnePassIteratorStaticSizes<uint32_t, CHUNK_N_PKTS, PACKET_SIZE_BYTES / sizeof(uint32_t)>;

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

FORCE_INLINE int32_t get_sender_num_unprocessed_acks(int32_t stream_id) { return get_ptr_val(stream_id); }

FORCE_INLINE void sender_mark_ack_processed(int32_t stream_id) { increment_local_update_ptr_val(stream_id, -1); }

FORCE_INLINE void send_ack_to_sender(int32_t stream_id) { remote_update_ptr_val<0>(stream_id, 1); }

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
            run_routing();
        }
    } else {
        while (erisc_info->channels[0].bytes_sent == 0) {
            run_routing();
        }
        eth_receiver_channel_done(0);
    }
}

template <typename OpenChunksCbT, typename SendPoolT>
void process_send_side(
    volatile TimingStats* timing_stats,
    // For now we can keep these flat?

    uint32_t local_src_ch_flow_control_stream_id,  // check this for new messages
    uint32_t local_src_ch_ack_stream_id,           // check this for acks from receiver
    uint64_t remote_src_ch_semaphore_ack_addr,     // send acks back here
    uint64_t remote_src_new_chunk_noc_addr,        // send new chunk info here

    chunk_forward_iterator_t& current_chunk_ptr,
    BufferIndex& current_chunk_ack_index,
    OpenChunksCbT& open_chunks_cb,
    SendPoolT& send_pool,
    uint32_t& unacked_sends,
    uint32_t& messages_sent,
    uint32_t& messages_acked,
    uint32_t messages_to_send,
    std::array<size_t, RX_N_PKTS>& receiver_buffer_addresses,
    tt::tt_fabric::ChannelBufferPointer<RX_N_PKTS>& sender_view_rx_buf_wr_ptr,
    uint32_t message_size) {
    // Process Ack

    // DPRINT << "Sender\n";
    // DPRINT << "\tsendpool empty: " << (uint32_t)send_pool.is_empty() << "\n";
    // DPRINT << "\tcurrent chunk ptr done: " << (uint32_t)current_chunk_ptr.is_done() << "\n";
    // DPRINT << "\topen chunks cb empty: " << (uint32_t)open_chunks_cb.is_empty() << "\n";

    auto n_unsent_messages = get_ptr_val(local_src_ch_flow_control_stream_id);
    if (n_unsent_messages > 0) {
        DPRINT << "ch " << (uint32_t)local_src_ch_flow_control_stream_id
               << " n_unsent_messages: " << (uint32_t)n_unsent_messages << "\n";
    }
    // Try to send from current chunk
    if (!eth_txq_is_busy() && n_unsent_messages > 0  // &&

        // if we got a chunk, it implies the worker already has a valid address, we don't need to check this
        // !open_chunks_cb.is_empty() &&

        // ditto as above
        // !current_chunk_ptr.is_done()
    ) {
        if (current_chunk_ptr.is_done()) {
            DPRINT << "ERROR CONDITION - CURRENT CHUNK IS EXHAUSTED BUT WE ARE TRYING TO SEND FROM IT\n";
            DPRINT << "\tcurrent_chunk_ptr.current_ptr: " << (uint32_t)(current_chunk_ptr.current_ptr) << "\n";
            DPRINT << "\tcurrent_chunk_ptr.end_ptr: " << (uint32_t)(current_chunk_ptr.end_ptr) << "\n";
            // while (1) {run_routing();};
        }

        ASSERT(!open_chunks_cb.is_empty()); // should not be possible to receive a message if this is false
        ASSERT(!current_chunk_ptr.is_done()); // should not be possible to receive a message if this is false

        auto current_buffer = open_chunks_cb.peek_front();

        // Fill packet data
        auto src_addr = current_chunk_ptr.get_current_ptr();
        reinterpret_cast<volatile uint32_t*>(src_addr)[0] = local_src_ch_ack_stream_id;
        DPRINT << "WRITE stream id(" << (uint32_t)local_src_ch_ack_stream_id << ") to " << (uint32_t)src_addr << "\n";
        auto dest_addr = receiver_buffer_addresses[sender_view_rx_buf_wr_ptr.get_buffer_index()];
        DPRINT << "ch " << (uint32_t)local_src_ch_flow_control_stream_id << " Sending packet from "
               << (uint32_t)src_addr << "\n\tto " << (uint32_t)dest_addr << "\n\tof size " << (uint32_t)message_size
               << "\n";

        // asm memory fence
        asm volatile("" ::: "memory");

        // Send packet using the buffer's channel ID
        eth_send_bytes_over_channel_payload_only_unsafe_one_packet((uint32_t)src_addr, dest_addr, message_size);

        DPRINT << "busy wait\n";
        while (eth_txq_is_busy()) {
        }
        DPRINT << "\tdone\n";
        notify_remote_receiver_of_new_packet();

        current_chunk_ptr.increment();

        increment_local_update_ptr_val(local_src_ch_flow_control_stream_id, -1);
        messages_sent++;

        sender_view_rx_buf_wr_ptr.increment();
        DPRINT << "\tnew rx wrptr: " << (uint32_t)sender_view_rx_buf_wr_ptr.get_buffer_index() << "\n";
        unacked_sends++;
    }

    // DPRINT << "chkpt 0\n";
    auto num_unprocessed_acks = get_sender_num_unprocessed_acks(local_src_ch_ack_stream_id);
    if (num_unprocessed_acks > 0) {
        DPRINT << "unprocessed acks\n";
        current_chunk_ack_index = BufferIndex{static_cast<uint8_t>(current_chunk_ack_index.get() + 1)};
        bool can_release_chunk = current_chunk_ack_index.get() == CHUNK_N_PKTS;
        uint64_t start_cycles;
        uint64_t end_cycles;
        if (can_release_chunk) {  // 22 cycles!??!?! (44 -> 22 if I don't include the branch condition in the timing)
            DPRINT << "Releasing chunk\n";
            start_cycles = eth_read_wall_clock();
            auto chunk = open_chunks_cb.pop();  // avg 7.3 cycles
            send_pool.return_chunk(chunk);      // 16 cycles
            current_chunk_ack_index = BufferIndex{0};
            end_cycles = eth_read_wall_clock();
        }
        sender_mark_ack_processed(local_src_ch_ack_stream_id);

        // Send completion ack (space available) to sender
        // noc_semaphore_inc(remote_src_ch_semaphore_ack_addr, 1);

        if (can_release_chunk) {
            timing_stats->release_count++;
            timing_stats->total_release_cycles += (end_cycles - start_cycles);
        }
        messages_acked++;
        unacked_sends--;
    }

    // Acquire new chunk if current chunk is full
    if ((open_chunks_cb.is_empty() || current_chunk_ptr.is_done()) && !send_pool.is_empty()) {
        DPRINT << "Acquiring new chunk\n";
        // Acquire new chunk - measure timing
        uint64_t start_cycles = eth_read_wall_clock();
        auto new_chunk = send_pool.get_free_chunk();

        // Currently we only support one-deep open chunks
        size_t new_chunk_base_address = new_chunk->get_buffer_address(BufferIndex{0});
        size_t new_chunk_field_for_sender = tt::tt_fabric::FabricChunkMessageAvailableMessage::pack(new_chunk_base_address);
        // notify the worker about new chunk availability
        DPRINT << "\tnotifying worker at " << (uint64_t)remote_src_new_chunk_noc_addr << " with new chunk value "
               << HEX() << (uint32_t)new_chunk_field_for_sender << " base addr " << (uint32_t)new_chunk_base_address
               << DEC() << "\n";
        noc_inline_dw_write(remote_src_new_chunk_noc_addr, new_chunk_field_for_sender);

        current_chunk_ptr.reset_to(reinterpret_cast<uint32_t*>(new_chunk_base_address));

        open_chunks_cb.push(new_chunk);
        uint64_t end_cycles = eth_read_wall_clock();

        timing_stats->total_acquire_cycles += (end_cycles - start_cycles);
        timing_stats->acquire_count++;
    }
    DPRINT << "end func\n";
}

void kernel_main() {
    // DPRINT << "Start\n";
    init_ptr_val(pkts_received_stream_id, 0);

    for (size_t i = 0; i < N_SRC_CHANS; i++) {
        init_ptr_val(i, 0);
        init_ptr_val(N_SRC_CHANS + i, 0);
    }

    DPRINT << "my_x: " << (uint32_t)my_x[0] << "\n";
    DPRINT << "my_y: " << (uint32_t)my_y[0] << "\n";

    uint32_t arg_idx = 0;
    const uint32_t handshake_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t total_messages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t message_size = get_arg_val<uint32_t>(arg_idx++);
    bool is_sender_offset_0 = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t send_buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t recv_buffer_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t timing_stats_addr = get_arg_val<uint32_t>(arg_idx++);

    // DPRINT << "Checkpoint 2\n";
    auto remote_src_noc_x_ords = reinterpret_cast<uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += N_SRC_CHANS;
    auto remote_src_noc_y_ords = reinterpret_cast<uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += N_SRC_CHANS;
    auto remote_src_semaphore_ack_addrs = reinterpret_cast<uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += N_SRC_CHANS;
    auto remote_src_new_chunk_addrs = reinterpret_cast<uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += N_SRC_CHANS;
    auto dest_noc_buffer_addrs = reinterpret_cast<uint32_t*>(get_arg_addr(arg_idx));
    arg_idx += N_SRC_CHANS;

    // DELETE ARGS
    // auto local_src_ch_semaphores = get_arg_val<uint32_t*>(arg_idx);
    // arg_idx += N_SRC_CHANS;

    // DPRINT << "Checkpoint 3 N_SRC_CHANS: " << (uint32_t)N_SRC_CHANS << "\n";
    using noc_addr_t = uint64_t;
    std::array<noc_addr_t, N_SRC_CHANS> remote_src_ch_semaphore_ack_addrs;
    std::array<noc_addr_t, N_SRC_CHANS> src_ch_new_chunk_addrs;
    std::array<noc_addr_t, N_SRC_CHANS> dest_noc_write_addrs;
    for (size_t i = 0; i < N_SRC_CHANS; i++) {
        remote_src_ch_semaphore_ack_addrs[i] = get_noc_addr(remote_src_noc_x_ords[i], remote_src_noc_y_ords[i], get_semaphore<ProgrammableCoreType::TENSIX>(remote_src_semaphore_ack_addrs[i]));
        src_ch_new_chunk_addrs[i] = get_noc_addr(remote_src_noc_x_ords[i], remote_src_noc_y_ords[i], get_semaphore<ProgrammableCoreType::TENSIX>(remote_src_new_chunk_addrs[i]));
        dest_noc_write_addrs[i] = get_noc_addr(remote_src_noc_x_ords[i], remote_src_noc_y_ords[i], dest_noc_buffer_addrs[i]);
    }

    // DPRINT << "Checkpoint 4\n";
    // Initialize timing stats at address provided by host
    timing_stats = reinterpret_cast<volatile TimingStats*>(timing_stats_addr);
    timing_stats->total_acquire_cycles = 0;
    timing_stats->total_release_cycles = 0;
    timing_stats->acquire_count = 0;
    timing_stats->release_count = 0;

    // Initialize separate channel buffer pools for send and receive

    // DPRINT << "Checkpoint 5\n";
    // Initialize pools dynamically based on N_CHUNKS
    std::array<size_t, N_CHUNKS> send_chunk_base_addresses;
    std::array<chunk_forward_iterator_t, N_SRC_CHANS> sender_channel_wrptrs;
    std::array<size_t, RX_N_PKTS> recv_buffer_addresses;

    for (uint32_t i = 0; i < N_CHUNKS; i++) {
        send_chunk_base_addresses[i] = send_buffer_base + i * PACKET_SIZE_BYTES * CHUNK_N_PKTS;
        DPRINT << "send_chunk_base_addresses[" << (uint32_t)i << "]: " << (uint32_t)send_chunk_base_addresses[i]
               << "\n";
        DPRINT << "\tPACKET_SIZE_BYTES: " << (uint32_t)PACKET_SIZE_BYTES << "\n";
        DPRINT << "\tCHUNK_N_PKTS: " << (uint32_t)CHUNK_N_PKTS << "\n";
        DPRINT << "\tsend_buffer_base: " << (uint32_t)send_buffer_base << "\n";
    }
    for (uint32_t i = 0; i < RX_N_PKTS; i++) {
        recv_buffer_addresses[i] = recv_buffer_base + i * PACKET_SIZE_BYTES;
        DPRINT << "recv_buffer_addresses[" << (uint32_t)i << "]: " << (uint32_t)recv_buffer_addresses[i] << "\n";
        DPRINT << "\tPACKET_SIZE_BYTES: " << (uint32_t)PACKET_SIZE_BYTES << "\n";
        DPRINT << "\trecv_buffer_base: " << (uint32_t)recv_buffer_base << "\n";
    }

    tt::tt_fabric::ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS> send_pool;
    send_pool.init(send_chunk_base_addresses, PACKET_SIZE_BYTES, sizeof(eth_channel_sync_t));

    // Set up channels for bidirectional communication if enabled
    const uint32_t messages_per_direction = N_SRC_CHANS == 0 ? total_messages : total_messages * N_SRC_CHANS;

    // Setup handshake
    // DPRINT << "Setup handshake\n";
    eth_setup_handshake(handshake_addr, is_sender_offset_0);
    // DPRINT << "Handshake done\n";

    uint32_t messages_acked = 0;
    uint32_t messages_sent = 0;
    uint32_t messages_received = 0;
    uint32_t total_messages_target = messages_per_direction;
    DPRINT << "total_messages_target: " << (uint32_t)total_messages_target << "\n";

    uint32_t idle_count = 0;
    const uint32_t idle_max = 100000;

#if defined(ARCH_WORMHOLE)
    // acq: 6.00
    // rel: 9.5
    std::array<
        tt::tt_fabric::WormholeEfficientCircularBuffer<typename tt::tt_fabric::ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS>::chunk_t*, N_CHUNKS>,
        N_SRC_CHANS>
        open_chunks_cbs;
#else
    // acq: 17
    // rel: 4
    std::array<
        tt::tt_fabric::CircularBuffer<typename tt::tt_fabric::ChannelBuffersPool<N_CHUNKS, CHUNK_N_PKTS>::chunk_t*, N_CHUNKS>,
        N_SRC_CHANS>
        open_chunks_cbs;
#endif


    tt::tt_fabric::ChannelBufferPointer<RX_N_PKTS> sender_view_rx_buf_wr_ptr;
    tt::tt_fabric::ChannelBufferPointer<RX_N_PKTS> receiver_rd_ptr;
    uint32_t unacked_sends = 0;
    uint32_t n_sender_channels_done = 0;
    std::array<bool, N_SRC_CHANS> completed_sender_channels;
    std::array<size_t, N_SRC_CHANS> sender_channels_acks_received;
    std::array<BufferIndex, N_SRC_CHANS> current_chunk_ack_indices;
    // DPRINT << "almost Main loop N_SRC_CHANS: " << (uint32_t)N_SRC_CHANS << "\n";
    for (size_t i = 0; i < N_SRC_CHANS; i++) {
        completed_sender_channels[i] = false;
        sender_channels_acks_received[i] = 0;
        current_chunk_ack_indices[i] = BufferIndex{0};
    }
    {
        DeviceZoneScopedN("FABRIC-ELASTIC-CHANNELS-TEST");

        auto test_start_cycles = eth_read_wall_clock();

        // DPRINT << "Main loop\n";

        while (n_sender_channels_done < N_SRC_CHANS || messages_received < total_messages_target) {
            bool made_progress = false;
            for (size_t i = 0; i < N_SRC_CHANS; i++) {
                // DPRINT << "loop\n";
                if (!completed_sender_channels[i]) {
                    process_send_side(
                        timing_stats,

                        i,                                     // check this stream autoinc registert for new messages
                        N_SRC_CHANS + i,                       // send acks back here
                        remote_src_ch_semaphore_ack_addrs[i],  // send acks back here
                        src_ch_new_chunk_addrs[i],             // send new chunk info here

                        sender_channel_wrptrs[i],
                        current_chunk_ack_indices[i],
                        open_chunks_cbs[i],
                        send_pool,
                        unacked_sends,
                        messages_sent,
                        messages_acked,
                        total_messages_target,

                        recv_buffer_addresses,
                        sender_view_rx_buf_wr_ptr,
                        message_size);

                    if (messages_acked >= total_messages_target) {
                        completed_sender_channels[i] = true;
                        n_sender_channels_done++;
                    }
                }

                // Put the receiver channel servicing here to keep the processing balanced
                if (BIDIRECTIONAL_MODE || !is_sender_offset_0) {
                    // Receiver logic - also use ChannelBuffersPool
                    auto num_unprocessed_packets = get_receiver_num_unprocessed_packets();
                    if (num_unprocessed_packets > 0 && !eth_txq_is_busy()) {
                        size_t packet_src_addr = recv_buffer_addresses[receiver_rd_ptr.get_buffer_index()];
                        DPRINT << "Receiving packet at " << (uint32_t)packet_src_addr << "\n";
                        auto remote_src_ack_stream_id = *reinterpret_cast<volatile uint32_t*>(packet_src_addr);
                        if (remote_src_ack_stream_id < N_SRC_CHANS || remote_src_ack_stream_id >= 2 * N_SRC_CHANS) {
                            DPRINT << "ERROR CONDITION - REMOTE SRC ACK STREAM ID IS OUT OF BOUNDS\n";
                            DPRINT << "\tremote_src_ack_stream_id: " << (uint32_t)remote_src_ack_stream_id << "\n";
                            while (1) {
                                run_routing();
                            };
                        }
                        receiver_mark_packet_processed();
                        messages_received++;
                        DPRINT << "\tSending ack to stream id(" << (uint32_t)remote_src_ack_stream_id << ")\n";
                        send_ack_to_sender(remote_src_ack_stream_id);

                        receiver_rd_ptr.increment();
                        made_progress = true;
                    }
                }
            }

            // Idle management
            if (made_progress) {
                idle_count = 0;
            } else {
                idle_count++;
                if (idle_count > idle_max) {
                    DPRINT << "Idle\n";
                    run_routing();
                    DPRINT << "Idle done\n";
                    idle_count = 0;
                }
            }
        }


        auto test_end_cycles = eth_read_wall_clock();
        timing_stats->total_test_cycles = test_end_cycles - test_start_cycles;
    }
}
